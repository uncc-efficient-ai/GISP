import os
from pathlib import Path

import yaml

from addict import Dict


def get_repository_paths():
    """Return the configurable filesystem roots used by the standalone repo."""
    default_root = Path(__file__).resolve().parents[2]
    repo_root = Path(os.environ.get('GISP_ROOT', default_root)).expanduser().resolve()
    data_root = Path(os.environ.get('GISP_DATA_ROOT', repo_root / 'data')).expanduser().resolve()
    output_root = Path(os.environ.get('GISP_OUTPUT_ROOT', repo_root / 'outputs')).expanduser().resolve()
    cache_root = Path(os.environ.get('GISP_CACHE_ROOT', data_root / 'cache')).expanduser().resolve()
    return {
        'repo_root': repo_root,
        'data_root': data_root,
        'output_root': output_root,
        'cache_root': cache_root,
    }


class Config:
    _instance = None

    class SafeLoaderWithJoin(yaml.SafeLoader):
        pass

    @classmethod
    def join_constructor(cls, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def __new__(cls, path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.config = None
        if path:
            cls._instance._load_config(path)
        return cls._instance

    @staticmethod
    def _resolve_string(value, paths):
        placeholders = {
            '${GISP_ROOT}': paths['repo_root'],
            '${GISP_DATA_ROOT}': paths['data_root'],
            '${GISP_OUTPUT_ROOT}': paths['output_root'],
            '${GISP_CACHE_ROOT}': paths['cache_root'],
        }
        for placeholder, replacement in placeholders.items():
            if placeholder in value:
                value = value.replace(placeholder, str(replacement))

        # Keep the published YAML files usable while their historical absolute
        # paths are migrated to placeholders. Longest prefixes must be checked
        # first so that output/data paths do not collapse to GISP_ROOT.
        legacy_prefixes = (
            ('/home/zwang53/gisp/denseLora', paths['output_root']),
            ('/root/gisp/denseLora', paths['output_root']),
            ('/home/zwang53/gisp/dataset', paths['data_root']),
            ('/root/gisp/dataset', paths['data_root']),
            ('/home/zwang53/gisp', paths['repo_root']),
            ('/root/gisp', paths['repo_root']),
        )
        for prefix, replacement in legacy_prefixes:
            if value == prefix or value.startswith(prefix + os.sep):
                value = str(replacement) + value[len(prefix):]
                break

        if '${GISP_' in value:
            raise ValueError(f'Unresolved GISP path placeholder: {value}')
        return value

    @classmethod
    def _resolve_paths(cls, value, paths):
        if isinstance(value, dict):
            return {key: cls._resolve_paths(item, paths) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._resolve_paths(item, paths) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._resolve_paths(item, paths) for item in value)
        if isinstance(value, str):
            return cls._resolve_string(value, paths)
        return value

    def _load_config(self, path):
        self.SafeLoaderWithJoin.add_constructor('!join', self.join_constructor)
        with open(path, 'r', encoding='utf-8') as file:
            config = yaml.load(file, Loader=self.SafeLoaderWithJoin)

        paths = get_repository_paths()
        self.config = Dict(self._resolve_paths(config, paths))
        self.config['config_path'] = str(Path(path).expanduser().resolve())

    def save_config(self, folder_path):
        if self.config is not None:
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(folder_path, 'config.yml')
            config_dict = self._convert_to_dict(self.config)
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False)

            return file_path
        else:
            raise ValueError("No configuration has been loaded yet")

    def _convert_to_dict(self, addict_obj):
        if isinstance(addict_obj, Dict):
            return {k: self._convert_to_dict(v) for k, v in addict_obj.items()}
        elif isinstance(addict_obj, list):
            return [self._convert_to_dict(item) for item in addict_obj]
        else:
            return addict_obj

    def get_config(self):
        return self.config

    def __getattr__(self, name):
        return getattr(self.config, name)


if __name__ == '__main__':
    config = Config('example_fuse.yml')
    c = config.get_config()
    print(c)
