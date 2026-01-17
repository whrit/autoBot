"""
Cost Model Integration Module.

Provides wrapper classes and factory functions for transaction cost modeling,
integrating the workspace cost_models library with CLI configuration options.

Supports four cost model types:
- ZeroCostModel: No transaction costs (for benchmarking)
- FixedCostModel: Fixed fee and slippage rates
- VolumeImpactCostModel: Volume-dependent slippage with market impact
- TieredCostModel: Tiered fee structure based on trade notional

Example usage:
    >>> from backtester_py.models.cost_model import create_cost_model, CostModelType
    >>> model = create_cost_model(
    ...     model_type=CostModelType.FIXED,
    ...     fee_rate=0.001,
    ...     slippage_rate=0.0005,
    ... )
    >>> fill_price = model.calculate_fill_price(
    ...     side="buy", ask_price=100.01, bid_price=100.00,
    ...     order_notional=10000, book_notional=500000, short_term_vol=0.001
    ... )

    # Using simplified Decimal-based API:
    >>> result = model.calculate_fill(
    ...     side="buy", price=Decimal("100.00"), quantity=Decimal("10"),
    ...     volume=Decimal("1000"), volatility=0.02
    ... )
    >>> print(f"Fill: {result.fill_price}, Commission: {result.commission}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal

# Import from workspace cost_models library
from cost_models import SlippageModel, TransactionCostModel


class CostModelType(str, Enum):
    """Available cost model types."""

    ZERO = "zero"
    FIXED = "fixed"
    VOLUME = "volume"
    TIERED = "tiered"


def _validate_inputs(
    side: str,
    price: Decimal,
    quantity: Decimal,
    volume: Decimal,
    volatility: float,
) -> None:
    """
    Validate common inputs for cost model calculations.

    Args:
        side: Order side ("buy" or "sell")
        price: Order price (must be positive)
        quantity: Order quantity (must be positive)
        volume: Market volume (must be non-negative)
        volatility: Market volatility (must be non-negative)

    Raises:
        ValueError: If any input is invalid
    """
    if side not in ("buy", "sell"):
        msg = f"Side must be 'buy' or 'sell', got '{side}'"
        raise ValueError(msg)
    if price <= 0:
        msg = f"Price must be positive, got {price}"
        raise ValueError(msg)
    if quantity <= 0:
        msg = f"Quantity must be positive, got {quantity}"
        raise ValueError(msg)
    if volume < 0:
        msg = f"Volume must be non-negative, got {volume}"
        raise ValueError(msg)
    if volatility < 0:
        msg = f"Volatility must be non-negative, got {volatility}"
        raise ValueError(msg)


@dataclass(frozen=True)
class FillResult:
    """
    Result of a fill price calculation.

    Supports both the legacy float API (slippage_bps, fee_bps, total_cost_bps)
    and the new Decimal API (commission, slippage).

    Attributes:
        fill_price: The calculated fill price after costs (Decimal or float)
        slippage_bps: Total slippage applied in basis points (legacy API)
        fee_bps: Fixed fee component in basis points (legacy API)
        total_cost_bps: Total cost (slippage + fee) in basis points (legacy API)
        commission: Commission charged on the trade (new Decimal API)
        slippage: Slippage cost in price terms (new Decimal API)
    """

    fill_price: float | Decimal
    slippage_bps: float = 0.0
    fee_bps: float = 0.0
    total_cost_bps: float = 0.0
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")


class CostModel(ABC):
    """
    Abstract base class for cost models.

    Defines the interface that all cost models must implement for
    compatibility with the backtester engine.
    """

    @property
    @abstractmethod
    def fixed_cost_bps(self) -> float:
        """Fixed cost component in basis points."""
        ...

    @abstractmethod
    def calculate_fill(
        self,
        side: Literal["buy", "sell"],
        price: Decimal,
        quantity: Decimal,
        volume: Decimal,
        volatility: float,
    ) -> FillResult:
        """
        Calculate fill price with slippage and commission (Decimal API).

        Args:
            side: Order side ("buy" or "sell")
            price: Base price for the order
            quantity: Order quantity
            volume: Market volume (for impact calculations)
            volatility: Market volatility

        Returns:
            FillResult containing fill price, commission, and slippage
        """
        ...

    @abstractmethod
    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate the fill price including all transaction costs.

        Args:
            side: Order side ("buy" or "sell")
            ask_price: Current best ask price
            bid_price: Current best bid price
            order_notional: Order size in notional terms
            book_notional: Top-of-book notional
            short_term_vol: Short-term realized volatility

        Returns:
            Expected fill price accounting for slippage and costs
        """
        ...

    def calculate_fill_with_details(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> FillResult:
        """
        Calculate fill price with detailed cost breakdown.

        Args:
            side: Order side ("buy" or "sell")
            ask_price: Current best ask price
            bid_price: Current best bid price
            order_notional: Order size in notional terms
            book_notional: Top-of-book notional
            short_term_vol: Short-term realized volatility

        Returns:
            FillResult with price and cost breakdown
        """
        fill_price = self.calculate_fill_price(
            side, ask_price, bid_price, order_notional, book_notional, short_term_vol
        )

        # Calculate slippage from fill price
        base_price = ask_price if side == "buy" else bid_price
        if base_price > 0:
            total_cost_bps = abs(fill_price - base_price) / base_price * 10000
        else:
            total_cost_bps = 0.0

        slippage_bps = max(0.0, total_cost_bps - self.fixed_cost_bps)

        return FillResult(
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            fee_bps=self.fixed_cost_bps,
            total_cost_bps=total_cost_bps,
        )


class ZeroCostModel(CostModel):
    """
    Zero cost model - no transaction costs applied.

    Useful for benchmarking or understanding raw strategy performance
    before accounting for realistic execution costs.

    Fills occur at the quoted price without any slippage or fees.
    """

    @property
    def fixed_cost_bps(self) -> float:
        """No fixed costs."""
        return 0.0

    def calculate_fill(
        self,
        side: Literal["buy", "sell"],
        price: Decimal,
        quantity: Decimal,
        volume: Decimal,
        volatility: float,
    ) -> FillResult:
        """Fill at quoted price without any costs (Decimal API)."""
        _validate_inputs(side, price, quantity, volume, volatility)
        return FillResult(
            fill_price=price,
            commission=Decimal("0"),
            slippage=Decimal("0"),
        )

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """Fill at quoted price without any costs."""
        return ask_price if side == "buy" else bid_price


class FixedCostModel(CostModel):
    """
    Fixed rate cost model with constant fee and slippage rates.

    Applies fixed percentage costs regardless of order size or market conditions.
    Simple and predictable, suitable for initial strategy evaluation.

    Attributes:
        fee_rate: Fixed fee rate as decimal (0.001 = 0.1% = 10 bps)
        slippage_rate: Fixed slippage rate as decimal (0.0005 = 0.05% = 5 bps)
        commission_rate: Alias for fee_rate (Decimal API)
    """

    def __init__(
        self,
        fee_rate: float | Decimal | None = None,
        slippage_rate: float | Decimal = 0.0005,
        *,
        commission_rate: Decimal | None = None,
    ) -> None:
        """
        Initialize the fixed cost model.

        Args:
            fee_rate: Fixed fee rate as decimal (default: 0.001 = 0.1%)
            slippage_rate: Fixed slippage rate as decimal (default: 0.0005 = 0.05%)
            commission_rate: Alias for fee_rate (Decimal API, takes precedence)
        """
        # Support both fee_rate and commission_rate
        if commission_rate is not None:
            self._commission_rate = Decimal(str(commission_rate))
            self._fee_rate = float(commission_rate)
        elif fee_rate is not None:
            self._fee_rate = float(fee_rate)
            self._commission_rate = Decimal(str(fee_rate))
        else:
            self._fee_rate = 0.001
            self._commission_rate = Decimal("0.001")

        self._slippage_rate_decimal = Decimal(str(slippage_rate))
        self._slippage_rate = float(slippage_rate)
        self._fee_bps = self._fee_rate * 10000
        self._slippage_bps = self._slippage_rate * 10000

    @property
    def fixed_cost_bps(self) -> float:
        """Fixed fee in basis points."""
        return self._fee_bps

    @property
    def fee_rate(self) -> float:
        """Fee rate as decimal."""
        return self._fee_rate

    @property
    def slippage_rate(self) -> float | Decimal:
        """Slippage rate as decimal."""
        return self._slippage_rate_decimal

    @property
    def commission_rate(self) -> Decimal:
        """Commission rate as Decimal (Decimal API)."""
        return self._commission_rate

    def calculate_fill(
        self,
        side: Literal["buy", "sell"],
        price: Decimal,
        quantity: Decimal,
        volume: Decimal,
        volatility: float,
    ) -> FillResult:
        """
        Calculate fill price with fixed costs (Decimal API).

        For buy orders: fill_price = price * (1 + slippage_rate)
        For sell orders: fill_price = price * (1 - slippage_rate)
        Commission = price * quantity * commission_rate
        """
        _validate_inputs(side, price, quantity, volume, volatility)

        # Calculate notional
        notional = price * quantity

        # Calculate commission on notional
        commission = notional * self._commission_rate

        # Calculate slippage in price terms
        slippage = price * self._slippage_rate_decimal

        # Apply slippage to price based on side
        if side == "buy":
            # Buy: pay more (price increases)
            fill_price = price * (Decimal("1") + self._slippage_rate_decimal)
        else:
            # Sell: receive less (price decreases)
            fill_price = price * (Decimal("1") - self._slippage_rate_decimal)

        return FillResult(
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
        )

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate fill price with fixed costs.

        Applies fixed slippage and fee rates regardless of order size.
        """
        total_cost_rate = self._fee_rate + self._slippage_rate

        if side == "buy":
            # Buy: pay ask price + costs
            return ask_price * (1 + total_cost_rate)
        else:
            # Sell: receive bid price - costs
            return bid_price * (1 - total_cost_rate)


class VolumeImpactCostModel(CostModel):
    """
    Volume-dependent cost model with market impact.

    Uses the workspace cost_models library for sophisticated slippage modeling
    based on spread, order size relative to book depth, and volatility.

    Formula:
        slippage_bps = spread_coef * spread
                     + size_coef * (order/book) * 100
                     + vol_coef * volatility * 100

    Also supports a simplified Decimal API with square-root market impact:
        base_impact = impact_coefficient * sqrt(quantity / volume)
        vol_adjustment = volatility_coefficient * volatility
        total_impact = min(base_impact + vol_adjustment, max_impact)

    Attributes:
        fee_rate: Fixed fee rate as decimal
        spread_coef: Coefficient for spread impact (default: 1.0)
        size_coef: Coefficient for order size impact (default: 1.0)
        vol_coef: Coefficient for volatility impact (default: 1.0)
        max_size_impact_bps: Maximum size impact cap (default: 50.0 bps)
        base_commission: Base commission rate (Decimal API)
        impact_coefficient: Volume impact coefficient (Decimal API)
        volatility_coefficient: Volatility coefficient (Decimal API)
        max_impact: Maximum impact rate (Decimal API)
    """

    def __init__(
        self,
        fee_rate: float | Decimal | None = None,
        spread_coef: float = 1.0,
        size_coef: float = 1.0,
        vol_coef: float = 1.0,
        max_size_impact_bps: float = 50.0,
        *,
        base_commission: Decimal | None = None,
        impact_coefficient: Decimal | None = None,
        volatility_coefficient: Decimal | None = None,
        max_impact: Decimal | None = None,
    ) -> None:
        """
        Initialize the volume impact cost model.

        Args:
            fee_rate: Fixed fee rate as decimal (default: 0.001 = 0.1%)
            spread_coef: Spread impact coefficient (default: 1.0)
            size_coef: Size impact coefficient (default: 1.0)
            vol_coef: Volatility impact coefficient (default: 1.0)
            max_size_impact_bps: Maximum size impact in bps (default: 50.0)
            base_commission: Base commission rate for Decimal API
            impact_coefficient: Volume impact coefficient for Decimal API
            volatility_coefficient: Volatility coefficient for Decimal API
            max_impact: Maximum impact cap for Decimal API
        """
        # Handle legacy float API
        if fee_rate is not None:
            self._fee_rate = float(fee_rate)
        elif base_commission is not None:
            self._fee_rate = float(base_commission)
        else:
            self._fee_rate = 0.001
        self._fee_bps = self._fee_rate * 10000

        # Decimal API parameters
        self._base_commission = (
            Decimal(str(base_commission)) if base_commission is not None
            else Decimal(str(self._fee_rate))
        )
        self._impact_coefficient = (
            Decimal(str(impact_coefficient)) if impact_coefficient is not None
            else Decimal("0.1")
        )
        self._volatility_coefficient = (
            Decimal(str(volatility_coefficient)) if volatility_coefficient is not None
            else Decimal("0.5")
        )
        self._max_impact = (
            Decimal(str(max_impact)) if max_impact is not None
            else Decimal("0.05")  # 5% max
        )

        # Create underlying cost_models instances for legacy API
        self._slippage_model = SlippageModel(
            spread_coef=spread_coef,
            size_coef=size_coef,
            vol_coef=vol_coef,
            max_size_impact_bps=max_size_impact_bps,
        )
        self._cost_model = TransactionCostModel(
            slippage_model=self._slippage_model,
            fixed_cost_bps=self._fee_bps,
        )

    @property
    def fixed_cost_bps(self) -> float:
        """Fixed fee in basis points."""
        return self._fee_bps

    @property
    def fee_rate(self) -> float:
        """Fee rate as decimal."""
        return self._fee_rate

    def calculate_fill(
        self,
        side: Literal["buy", "sell"],
        price: Decimal,
        quantity: Decimal,
        volume: Decimal,
        volatility: float,
    ) -> FillResult:
        """
        Calculate fill with volume-based market impact (Decimal API).

        Uses square-root market impact model:
        - Impact proportional to sqrt(participation_rate)
        - Volatility adds additional impact
        - Total impact capped at max_impact
        """
        import math

        _validate_inputs(side, price, quantity, volume, volatility)

        # Calculate notional for commission
        notional = price * quantity

        # Calculate commission
        commission = notional * self._base_commission

        # Calculate participation rate
        if volume > 0:
            participation = quantity / volume
            # Square-root market impact model
            sqrt_participation = Decimal(str(math.sqrt(float(participation))))
            base_impact = self._impact_coefficient * sqrt_participation
        else:
            # Zero volume: use maximum impact
            base_impact = self._max_impact

        # Volatility adjustment
        vol_impact = self._volatility_coefficient * Decimal(str(volatility))

        # Total impact (capped)
        total_impact = min(base_impact + vol_impact, self._max_impact)

        # Calculate slippage in price terms
        slippage = price * total_impact

        # Apply to price based on side
        if side == "buy":
            fill_price = price * (Decimal("1") + total_impact)
        else:
            fill_price = price * (Decimal("1") - total_impact)

        return FillResult(
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
        )

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate fill price with volume-dependent costs.

        Delegates to the cost_models library for sophisticated slippage calculation.
        """
        result = self._cost_model.calculate_fill_price(
            side=side,
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )
        return float(result)

    def calculate_round_trip_cost_bps(
        self,
        spread_bps: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate total round-trip cost in basis points.

        Useful for evaluating strategy profitability thresholds.

        Args:
            spread_bps: Current bid-ask spread in basis points
            order_notional: Order size in notional terms
            book_notional: Top-of-book notional
            short_term_vol: Short-term realized volatility

        Returns:
            Total round-trip cost in basis points
        """
        result = self._cost_model.calculate_round_trip_cost_bps(
            spread_bps=spread_bps,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )
        return float(result)


class TieredCostModel(CostModel):
    """
    Tiered fee structure based on trade notional value.

    Implements exchange-like tiered fee structures where larger
    trades receive lower commission rates.

    Each tier is defined as (threshold, rate) where:
    - threshold: Minimum notional to qualify for this tier
    - rate: Commission rate for this tier

    Tiers must be sorted by threshold in ascending order.
    The highest qualifying tier is used.

    Attributes:
        tiers: List of (threshold, rate) tuples defining fee tiers
        slippage_rate: Fixed slippage rate applied to all orders

    Example:
        >>> tiers = [
        ...     (Decimal("0"), Decimal("0.001")),       # 0.1% for < $10k
        ...     (Decimal("10000"), Decimal("0.0008")),  # 0.08% for $10k-$100k
        ...     (Decimal("100000"), Decimal("0.0005")), # 0.05% for >= $100k
        ... ]
        >>> model = TieredCostModel(tiers=tiers, slippage_rate=Decimal("0.0003"))
    """

    def __init__(
        self,
        tiers: list[tuple[Decimal, Decimal]],
        slippage_rate: Decimal = Decimal("0.0003"),
    ) -> None:
        """
        Initialize the tiered cost model.

        Args:
            tiers: List of (threshold, rate) tuples
            slippage_rate: Fixed slippage rate (default: 0.0003 = 0.03%)

        Raises:
            ValueError: If tiers is empty
        """
        if not tiers:
            msg = "At least one tier must be defined"
            raise ValueError(msg)

        # Ensure tiers are sorted by threshold
        self._tiers = sorted(tiers, key=lambda t: t[0])
        self._slippage_rate = Decimal(str(slippage_rate))
        # For legacy API compatibility, use lowest tier rate as fixed cost
        self._fee_bps = float(self._tiers[0][1]) * 10000

    @property
    def fixed_cost_bps(self) -> float:
        """Base fixed cost in basis points (lowest tier rate)."""
        return self._fee_bps

    def _get_commission_rate(self, notional: Decimal) -> Decimal:
        """
        Get the commission rate for a given notional value.

        Finds the highest tier threshold that the notional meets or exceeds
        and returns the corresponding rate.

        Args:
            notional: Trade notional value

        Returns:
            Commission rate for the applicable tier
        """
        rate = self._tiers[0][1]  # Default to lowest tier

        for threshold, tier_rate in self._tiers:
            if notional >= threshold:
                rate = tier_rate
            else:
                break

        return rate

    def calculate_fill(
        self,
        side: Literal["buy", "sell"],
        price: Decimal,
        quantity: Decimal,
        volume: Decimal,
        volatility: float,
    ) -> FillResult:
        """
        Calculate fill with tiered commission structure (Decimal API).

        Commission rate is determined by the trade notional value.
        Slippage is applied at a fixed rate.
        """
        _validate_inputs(side, price, quantity, volume, volatility)

        # Calculate notional
        notional = price * quantity

        # Get tiered commission rate
        commission_rate = self._get_commission_rate(notional)

        # Calculate commission
        commission = notional * commission_rate

        # Calculate slippage
        slippage = price * self._slippage_rate

        # Apply slippage to price based on side
        if side == "buy":
            fill_price = price * (Decimal("1") + self._slippage_rate)
        else:
            fill_price = price * (Decimal("1") - self._slippage_rate)

        return FillResult(
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
        )

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate fill price with tiered costs (legacy float API).

        Commission rate determined by order notional, slippage is fixed.
        """
        notional = Decimal(str(order_notional))
        commission_rate = float(self._get_commission_rate(notional))
        slippage_rate = float(self._slippage_rate)

        total_cost_rate = commission_rate + slippage_rate

        if side == "buy":
            return ask_price * (1 + total_cost_rate)
        else:
            return bid_price * (1 - total_cost_rate)


def create_cost_model(
    model_type: str | CostModelType = CostModelType.FIXED,
    fee_rate: float | Decimal = 0.001,
    slippage_rate: float | Decimal = 0.0005,
    spread_coef: float = 1.0,
    size_coef: float = 1.0,
    vol_coef: float = 1.0,
    max_size_impact_bps: float = 50.0,
    *,
    tiers: list[tuple[Decimal, Decimal]] | None = None,
) -> CostModel:
    """
    Factory function to create a cost model based on type.

    Args:
        model_type: Type of cost model ("zero", "fixed", "volume", "tiered")
        fee_rate: Fixed fee rate as decimal (for fixed/volume models)
        slippage_rate: Fixed slippage rate as decimal (for fixed/tiered model)
        spread_coef: Spread coefficient (for volume model)
        size_coef: Size coefficient (for volume model)
        vol_coef: Volatility coefficient (for volume model)
        max_size_impact_bps: Max size impact cap (for volume model)
        tiers: Fee tiers for tiered model (list of (threshold, rate) tuples)

    Returns:
        Configured CostModel instance

    Raises:
        ValueError: If model_type is invalid
    """
    if isinstance(model_type, str):
        model_type = CostModelType(model_type.lower())

    if model_type == CostModelType.ZERO:
        return ZeroCostModel()
    elif model_type == CostModelType.FIXED:
        return FixedCostModel(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
    elif model_type == CostModelType.VOLUME:
        return VolumeImpactCostModel(
            fee_rate=fee_rate,
            spread_coef=spread_coef,
            size_coef=size_coef,
            vol_coef=vol_coef,
            max_size_impact_bps=max_size_impact_bps,
        )
    elif model_type == CostModelType.TIERED:
        if tiers is None:
            # Default tiers if none provided
            tiers = [
                (Decimal("0"), Decimal("0.001")),
                (Decimal("10000"), Decimal("0.0008")),
                (Decimal("100000"), Decimal("0.0005")),
                (Decimal("1000000"), Decimal("0.0002")),
            ]
        return TieredCostModel(
            tiers=tiers,
            slippage_rate=Decimal(str(slippage_rate)),
        )
    else:
        raise ValueError(f"Unknown cost model type: {model_type}")


__all__ = [
    "CostModel",
    "CostModelType",
    "FillResult",
    "FixedCostModel",
    "TieredCostModel",
    "VolumeImpactCostModel",
    "ZeroCostModel",
    "create_cost_model",
]
