"""This module loads pylelemmatize and torch, and re-exports all pylelemmatize symbols.
"""
import torch
import pylelemmatize
from pylelemmatize import * 
__all__ = pylelemmatize.__all__