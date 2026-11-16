from pathlib import Path
import json
import numpy as np
import torch
from csp_sca.cli import main
from test_pipeline import prepare_tasks, small_config


def test_documented_commands_run_with_prepared_cpu_fixture(tmp_path, capsys):
    torch.set_num_threads(2)
    data = tmp_path / 'data'; prepare_tasks(data)
    config = tmp_path / 'config.json'; config.write_text(json.dumps(small_config()))
    run = tmp_path / 'run'
    main(['validate-data', '--config', str(config), '--data', str(data)])
    main(['train', '--config', str(config), '--data', str(data), '--output', str(run),
          '--method', 'unprotected', '--seed', '13', '--device', 'cpu'])
    main(['evaluate', '--run', str(run), '--data', str(data), '--device', 'cpu'])
    main(['report', '--runs', str(run), '--output', str(tmp_path / 'reports')])
    assert (tmp_path / 'reports/summary.csv').is_file()
    main(['plot', 'nicv', '--config', str(config), '--data', str(data), '--output', str(tmp_path / 'nicv.pdf')])
    main(['plot', 'ge', '--runs', str(run), '--output', str(tmp_path / 'ge.pdf')])
    for stage, name in [(0, 'before'), (1, 'after')]:
        main(['features', '--run', str(run), '--data', str(data), '--task', 'F0',
              '--stage', str(stage), '--samples', '8', '--device', 'cpu', '--output', str(tmp_path / (name + '.npz'))])
    drift = tmp_path / 'drift.json'
    main(['drift', '--before', str(tmp_path / 'before.npz'), '--after', str(tmp_path / 'after.npz'),
          '--label', 'Unprotected', '--output', str(drift)])
    main(['plot', 'retention', '--runs', str(run), '--drift', str(drift), '--output', str(tmp_path / 'retention.pdf')])
    for name in ['nicv.pdf', 'ge.pdf', 'retention.pdf']:
        assert (tmp_path / name).read_bytes().startswith(b'%PDF')


def test_main_suite_has_all_seeds_and_does_not_start_jobs_in_dry_run(capsys):
    recipe = Path(__file__).resolve().parents[1] / 'configs/suites.json'
    main(['suite', '--recipe', str(recipe), '--name', 'main', '--dry-run'])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 21
    assert all(any('seed' + str(seed) in row for row in lines) for seed in [13, 42, 2026])
