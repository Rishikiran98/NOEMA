"""M1.2 regression tests: coder-model presets, RAM fit, and thinking control (audit G4).

Model sizes and shapes below are the preset's recorded values, not measurements; real
tokens/s and RAM use are NOT RUN until the user runs a preset on the laptop.
"""
import contextlib
import io
import json
import unittest
from tempfile import TemporaryDirectory

from cpu_first.core import LocalLlama, hybrid_thinking
from cpu_platform import hardware, launch
from cpu_platform.hardware import GB, MODEL_PRESETS
from tests.fake_llama import FakeLlama
from tests.test_v052 import FULL_HELP

CODER = "qwen3-coder-30b-a3b"
DENSE = "qwen2.5-coder-7b"


def fake_hardware(total_gb: float):
    real = launch.detect

    def detect(measure_bandwidth=True):
        info = real(False)
        total = int(total_gb * GB)
        return {**info, "memory": {"total_bytes": total, "available_bytes": None,
                                   "total_gb": round(total / GB, 1)},
                "recommended": hardware.recommend(total, 6, 12, None)}
    return detect


class CoderPresets(unittest.TestCase):
    def test_presets_declare_role_thinking_and_shape(self):
        coder, dense = MODEL_PRESETS[CODER], MODEL_PRESETS[DENSE]
        for preset in (coder, dense):
            self.assertEqual(preset.role, "coder")
            self.assertFalse(preset.thinking)
            self.assertEqual(preset.ctx_size, 16384)
        self.assertEqual(coder.active_params_b, 3.3)
        self.assertIsNone(dense.active_params_b)
        self.assertEqual(coder.kv_bytes_per_token, 98304)   # 2 * 48 layers * 4 KV heads * 128 * 2 B
        self.assertEqual(dense.kv_bytes_per_token, 57344)   # 2 * 28 * 4 * 128 * 2 B
        self.assertTrue(coder.hf.endswith(":Q4_K_M") and dense.hf.endswith(":Q4_K_M"))
        # Every general preset keeps hybrid thinking and its old defaults.
        for name in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"):
            self.assertEqual((MODEL_PRESETS[name].role, MODEL_PRESETS[name].thinking), ("general", True))

    def test_moe_reads_only_active_weights_per_token(self):
        coder, dense = MODEL_PRESETS[CODER], MODEL_PRESETS[DENSE]
        self.assertLess(coder.active_bytes, 0.15 * coder.approx_bytes)
        self.assertEqual(dense.active_bytes, dense.approx_bytes)
        est = hardware.recommend(32 * GB, 6, 12, 20e9)["token_rate_estimate_per_s"]
        # Larger file, but fewer bytes per token: the MoE estimate beats the dense 7B.
        self.assertGreater(est[CODER], est[DENSE])


class RamFit(unittest.TestCase):
    def test_budget_is_max_of_fraction_and_fixed_reserve(self):
        self.assertAlmostEqual(hardware.ram_budget(32 * GB) / GB, 22.0)
        self.assertAlmostEqual(hardware.ram_budget(16 * GB) / GB, 9.6)
        self.assertAlmostEqual(hardware.ram_budget(8 * GB) / GB, 4.8)

    def test_coder_fits_32gb_laptop_not_16gb(self):
        coder = MODEL_PRESETS[CODER]
        need = hardware.model_memory_bytes(coder, 16384) / GB
        self.assertAlmostEqual(need, 18.6 + 0.5 + 1.5, places=1)
        # Windows typically reports a little under 32 GiB for 32 GB installed.
        self.assertTrue(hardware.model_fits(coder, int(31.7 * GB), 16384))
        self.assertFalse(hardware.model_fits(coder, 16 * GB, 16384))
        self.assertFalse(hardware.model_fits(coder, 16 * GB, 2048))

    def test_dense_fallback_fits_16gb_not_8gb(self):
        dense = MODEL_PRESETS[DENSE]
        self.assertTrue(hardware.model_fits(dense, 16 * GB, 16384))
        self.assertFalse(hardware.model_fits(dense, 8 * GB, 16384))

    def test_recommend_picks_coder_by_ram_and_keeps_large_general_model(self):
        self.assertEqual(hardware.recommend(32 * GB, 6, 12, None)["coder_model"], CODER)
        self.assertEqual(hardware.recommend(32 * GB, 6, 12, None)["large_model"], "qwen3-8b")
        self.assertEqual(hardware.recommend(16 * GB, 4, 8, None)["coder_model"], DENSE)
        self.assertIsNone(hardware.recommend(8 * GB, 4, 8, None)["coder_model"])

    def test_unknown_ram_only_allows_small_default(self):
        self.assertFalse(hardware.model_fits(MODEL_PRESETS[DENSE], None, 16384))


class ThinkingControl(unittest.TestCase):
    def test_hybrid_thinking_names(self):
        for name in ("Qwen3-0.6B-Q8_0.gguf", "qwen3-8b", "Qwen3-30B-A3B-Q4_K_M.gguf"):
            self.assertTrue(hybrid_thinking(name), name)
        for name in ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "qwen3-coder-30b-a3b",
                     "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", "Qwen3-4B-Thinking-2507",
                     "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "llama-3.1-8b", ""):
            self.assertFalse(hybrid_thinking(name), name)

    def test_coder_requests_carry_no_thinking_switches(self):
        fake = FakeLlama("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", accuracy=1.0, think_capable=False)
        try:
            client = LocalLlama(fake.url)
            self.assertFalse(client.disables_thinking)
            client.chat("Answer.", "What does this function return?")
            body = fake.requests[-1]
            self.assertEqual(body["messages"][0]["content"], "Answer.")
            self.assertNotIn("chat_template_kwargs", body)
            self.assertNotIn("reasoning_budget_tokens", body)
        finally:
            fake.close()

    def test_hybrid_qwen3_unchanged(self):
        fake = FakeLlama("Qwen3-0.6B-Q8_0.gguf", accuracy=1.0)
        try:
            LocalLlama(fake.url).chat("Solve.", "x")
            body = fake.requests[-1]
            self.assertTrue(body["messages"][0]["content"].endswith("/no_think"))
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        finally:
            fake.close()

    def test_coder_command_has_jinja_but_no_reasoning(self):
        cmd, _ = launch.build_command(["llama-server"], FULL_HELP,
                                      model_source=("hf", MODEL_PRESETS[CODER].hf), port=8080,
                                      threads=6, ctx_size=16384, allow_gpu=False,
                                      is_qwen3=True, thinking=False)
        self.assertIn("--jinja", cmd)
        self.assertNotIn("--reasoning", cmd)
        self.assertEqual(cmd[cmd.index("--ctx-size") + 1], "16384")
        # Default keeps the v0.8 behaviour for hybrid Qwen3.
        cmd, _ = launch.build_command(["llama-server"], FULL_HELP, model_source=("hf", "x"), port=1,
                                      threads=1, ctx_size=4096, allow_gpu=False, is_qwen3=True)
        self.assertIn("auto", cmd[cmd.index("--reasoning") + 1])


class LauncherCoder(unittest.TestCase):
    def run_launch(self, total_gb, *argv):
        original = launch.detect
        launch.detect = fake_hardware(total_gb)
        out = io.StringIO()
        try:
            with TemporaryDirectory() as tmp, contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = launch.main([*argv, "--results-dir", tmp])
        finally:
            launch.detect = original
        return code, out.getvalue()

    def manifest(self, text):
        return json.loads(text[text.index("\n{") + 1:])

    def test_dry_run_coder_on_32gb(self):
        code, out = self.run_launch(31.7, "--model", CODER, "--dry-run")
        self.assertEqual(code, 0)
        command = next(line for line in out.splitlines() if line.startswith("Command:"))
        self.assertIn("--ctx-size 16384", command)
        self.assertNotIn("--reasoning", command)
        model = self.manifest(out)["model"]
        self.assertEqual((model["role"], model["hybrid_thinking"], model["active_params_b"]),
                         ("coder", False, 3.3))
        self.assertGreater(model["estimated_memory_bytes"], 20 * GB)

    def test_coder_refused_on_16gb_even_with_small_context(self):
        for extra in ((), ("--ctx-size", "2048")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.run_launch(16, "--model", CODER, "--dry-run", *extra)

    def test_explicit_ctx_size_overrides_preset(self):
        code, out = self.run_launch(31.7, "--model", DENSE, "--dry-run", "--ctx-size", "8192")
        self.assertEqual(code, 0)
        self.assertIn("--ctx-size 8192", out)

    def test_general_preset_keeps_hardware_context_and_reasoning(self):
        code, out = self.run_launch(31.7, "--model", "qwen3-0.6b", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("--ctx-size 4096", out)
        self.assertTrue(self.manifest(out)["model"]["hybrid_thinking"])

    def test_list_models_shows_coder_presets(self):
        code, out = self.run_launch(31.7, "--list-models")
        self.assertEqual(code, 0)
        coder_line = next(line for line in out.splitlines() if line.startswith(CODER))
        self.assertIn("3.3B act", coder_line)
        self.assertIn(" fits", coder_line)
        self.assertIn(DENSE, out)
        self.assertIn(MODEL_PRESETS[CODER].hf, out)


if __name__ == "__main__":
    unittest.main()
