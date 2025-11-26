from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Union, Callable
from ase.calculators.calculator import Calculator, all_changes
from pathlib import Path

class CalculatorWrapper(ABC):
    """Registry for calculator types and instances."""

    _calculators: Dict[str, Type['CalculatorWrapper']] = {}

    @classmethod
    def get_calculator(cls, name: str):
        """Get a registered calculator by name."""
        try:
            return cls._calculators[name]
        except KeyError as e:
            raise RuntimeError("unknown calculator: " + name) from e

    @classmethod
    def get_all_calculator(cls) -> list:
        """List all registered calculators."""
        return list(cls._calculators.keys())
    
    @classmethod
    def register(cls,name: str):
        """Decorator for registering calculators."""
        def decorator(calculator):
            cls._calculators[name] = calculator
            return calculator
        return decorator
    
    @abstractmethod
    def create(self, model_path: Optional[Union[str, Path]] = None, **kwargs) -> Calculator:
        pass
    

@CalculatorWrapper.register('mattersim')
class MattersimCalculatorWrapper(CalculatorWrapper):
    """MatterSim calculator wrapper."""
    def create(self, model_path: str, **kwargs) -> Calculator:
        """Create MatterSim calculator."""
        try:
            from mattersim.forcefield import MatterSimCalculator
            return MatterSimCalculator.from_checkpoint(load_path=model_path, **kwargs)
        except ImportError as e:
            raise ImportError("MatterSim not available. Install with: pip install mattersim") from e
    

@CalculatorWrapper.register('deepmd')
@CalculatorWrapper.register('dp')
class DPCalculatorWrapper(CalculatorWrapper):
    """DeepMD calculator wrapper."""
    def create(self, model_path:str, **kwargs) -> Calculator:
        """Create DeepMD calculator."""
        try:
            from deepmd.calculator import DP
            return DP(model=model_path,**kwargs)
        except ImportError as e:
            raise ImportError("DeepMD not available. Install with: pip install deepmd-kit") from e

@CalculatorWrapper.register('mace')
class MACECalculatorWrapper(CalculatorWrapper):
    """MACE calculator wrapper."""
    def create(self, model_path: str, **kwargs) -> Calculator:
        """Create MACE calculator."""
        try:
            from mace.calculators import MACECalculator
            return MACECalculator(model_paths=model_path, **kwargs)
        except ImportError as e:
            raise ImportError("MACE not available. Install with: pip install mace") from e


@CalculatorWrapper.register('emt')
class EMTCalculatorWrapper(CalculatorWrapper):
    """EMT calculator wrapper for testing."""
    def create(self, model_path: Optional[Union[str, Path]] = None, **kwargs) -> Calculator:
        """Create EMT calculator (no model path needed)."""
        try:
            from ase.calculators.emt import EMT
            return EMT(**kwargs)
        except ImportError as e:
            raise ImportError("EMT calculator not available") from e


@CalculatorWrapper.register('lj')
class LennardJonesCalculatorWrapper(CalculatorWrapper):
    """Lennard-Jones calculator wrapper for testing."""
    def create(self, model_path: Optional[Union[str, Path]] = None, **kwargs) -> Calculator:
        """Create Lennard-Jones calculator (no model path needed)."""
        try:
            from ase.calculators.lj import LennardJones
            return LennardJones(**kwargs)
        except ImportError as e:
            raise ImportError("LennardJones calculator not available") from e