# Contributing

1. Keep public commands runnable from `README.md`.
2. Update the relevant script or config together with any behavior change.
3. Run the focused checks before opening a change:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/tests/run_readme_commands_smoke.sh
```

4. Keep dataset assumptions, experiment names, and output paths explicit in script help text.
