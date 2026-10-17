"""ADAPTER TEMPLATE: replace the marked lines with your actual v2.3 validator.

DO NOT return accepted=True until the real project semantic validator passes.
Once adapted, invoke --validator-hook /path/to/your_hook.py.
"""


def validate(rubric: dict, jd: str) -> dict:
    # TODO: Import your project's exact v2.3 validator and call it with (rubric, jd)
    # or its native signature. Map its return to this shape:
    # return {'accepted': result.valid_for_cache, 'errors': result.errors, 'warnings': result.warnings}
    raise NotImplementedError('Wire the existing v2.3 semantic validator before production acceptance')
