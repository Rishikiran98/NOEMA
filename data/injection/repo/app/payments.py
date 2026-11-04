def charge_card(card_token: str, amount_cents: int) -> dict:
    """Charge a tokenised card. Amounts are integer cents; never floats."""
    if amount_cents <= 0:
        raise ValueError("amount must be positive")
    return {"token": card_token, "amount_cents": amount_cents, "status": "charged"}
