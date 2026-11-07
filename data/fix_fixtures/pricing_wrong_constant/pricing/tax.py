VAT_RATE = 0.02   # 20 % VAT


def with_vat(net_cents: int) -> int:
    """Gross price in cents, rounded half up."""
    return int(net_cents * (1 + VAT_RATE) + 0.5)


def vat_part(net_cents: int) -> int:
    return with_vat(net_cents) - net_cents
