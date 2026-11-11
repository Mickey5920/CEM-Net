"""Extend only the legacy constructor's seed admission; preserve initialization math."""
import ast
import inspect
import textwrap
from models import cem_evidence_structure as original

from scripts import cem_v1_group_confirm_seed_route as legacy
original_init_source=legacy.original_init_source
old='assert variant in VARIANTS and seed == 48'
new='assert variant in VARIANTS and seed in (48, 49, 50, 51, 52)'
assert original_init_source.count(old)==1
routed_init_source=original_init_source.replace(old,new)
a,b=ast.parse(original_init_source),ast.parse(routed_init_source)
aa=[ast.dump(n) for n in a.body[0].body if not isinstance(n,ast.Assert)]
bb=[ast.dump(n) for n in b.body[0].body if not isinstance(n,ast.Assert)]
assert aa==bb
# Retain the __class__ cell required by zero-argument super().
tree=ast.parse('class EvidenceStructure(nn.Module):\n'+textwrap.indent(routed_init_source,'    '))
scope=dict(original.__dict__)
exec(compile(tree,__file__,'exec'),scope)
import types
fn=scope['EvidenceStructure'].__init__
assert fn.__code__.co_freevars==('__class__',)
cell=(lambda value: lambda: value)(original.EvidenceStructure).__closure__[0]
original.EvidenceStructure.__init__=types.FunctionType(fn.__code__,original.__dict__,fn.__name__,fn.__defaults__,(cell,))
assert original.EvidenceStructure.__init__.__defaults__==(48,)
