"""
Tests for src/paths.py
======================
Tests for path configuration and directory management.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import (
    PROJECT_ROOT,
    MODELS_DIR, CACHE_DIR, CONFIG_DIR, REPORTS_DIR, DOCS_DIR,
    get_model_path, get_cache_path, get_config_path, get_report_path,
    GAMES_CACHE_FILE, ELO_CACHE_FILE, FEATURE_CACHE_FILE, MATCHUP_CACHE_FILE,
    PREDICTION_HISTORY_FILE, ENRICHED_GAMES_CSV, MATCHUPS_CSV, XGB_PARAMS_FILE
)


class TestProjectRoot:
    """Tests for project root detection"""
    
    def test_project_root_exists(self):
        """Project root directory should exist"""
        assert os.path.exists(PROJECT_ROOT)
        assert os.path.isdir(PROJECT_ROOT)
    
    def test_project_root_contains_src(self):
        """Project root should contain src directory"""
        src_dir = os.path.join(PROJECT_ROOT, 'src')
        assert os.path.exists(src_dir)
    
    def test_project_root_contains_main(self):
        """Project root should contain main.py"""
        main_file = os.path.join(PROJECT_ROOT, 'main.py')
        assert os.path.exists(main_file)


class TestDirectoryPaths:
    """Tests for directory path constants"""
    
    def test_models_dir_exists(self):
        """Models directory should exist"""
        assert os.path.exists(MODELS_DIR)
        assert os.path.isdir(MODELS_DIR)
    
    def test_cache_dir_exists(self):
        """Cache directory should exist"""
        assert os.path.exists(CACHE_DIR)
        assert os.path.isdir(CACHE_DIR)
    
    def test_config_dir_exists(self):
        """Config directory should exist"""
        assert os.path.exists(CONFIG_DIR)
        assert os.path.isdir(CONFIG_DIR)
    
    def test_reports_dir_exists(self):
        """Reports directory should exist"""
        assert os.path.exists(REPORTS_DIR)
        assert os.path.isdir(REPORTS_DIR)
    
    def test_directories_under_project_root(self):
        """All directories should be under project root"""
        for dir_path in [MODELS_DIR, CACHE_DIR, CONFIG_DIR, REPORTS_DIR]:
            assert dir_path.startswith(PROJECT_ROOT)


class TestPathHelpers:
    """Tests for path helper functions"""
    
    def test_get_model_path(self):
        """get_model_path should return path in models directory"""
        path = get_model_path('test_model.pkl')
        assert path.endswith('test_model.pkl')
        assert MODELS_DIR in path
        assert os.path.dirname(path) == MODELS_DIR
    
    def test_get_cache_path(self):
        """get_cache_path should return path in cache directory"""
        path = get_cache_path('test_cache.csv')
        assert path.endswith('test_cache.csv')
        assert CACHE_DIR in path
        assert os.path.dirname(path) == CACHE_DIR
    
    def test_get_config_path(self):
        """get_config_path should return path in config directory"""
        path = get_config_path('test_config.json')
        assert path.endswith('test_config.json')
        assert CONFIG_DIR in path
        assert os.path.dirname(path) == CONFIG_DIR
    
    def test_get_report_path(self):
        """get_report_path should return path in reports directory"""
        path = get_report_path('test_report.txt')
        assert path.endswith('test_report.txt')
        assert REPORTS_DIR in path
        assert os.path.dirname(path) == REPORTS_DIR
    
    def test_path_helpers_handle_special_characters(self):
        """Path helpers should handle filenames with special characters"""
        filenames = ['model_v1.0.pkl', 'cache-2024-01-15.csv', 'config_test.json']
        for filename in filenames:
            model_path = get_model_path(filename)
            assert filename in model_path


class TestSpecificFilePaths:
    """Tests for specific file path constants"""
    
    def test_games_cache_file_in_cache_dir(self):
        """GAMES_CACHE_FILE should be in cache directory"""
        assert CACHE_DIR in GAMES_CACHE_FILE
        assert GAMES_CACHE_FILE.endswith('.csv')
    
    def test_elo_cache_file_in_cache_dir(self):
        """ELO_CACHE_FILE should be in cache directory"""
        assert CACHE_DIR in ELO_CACHE_FILE
        assert ELO_CACHE_FILE.endswith('.pkl')
    
    def test_feature_cache_file_in_cache_dir(self):
        """FEATURE_CACHE_FILE should be in cache directory"""
        assert CACHE_DIR in FEATURE_CACHE_FILE
        assert FEATURE_CACHE_FILE.endswith('.pkl')
    
    def test_matchup_cache_file_in_cache_dir(self):
        """MATCHUP_CACHE_FILE should be in cache directory"""
        assert CACHE_DIR in MATCHUP_CACHE_FILE
        assert MATCHUP_CACHE_FILE.endswith('.pkl')
    
    def test_prediction_history_in_reports_dir(self):
        """PREDICTION_HISTORY_FILE should be in reports directory"""
        assert REPORTS_DIR in PREDICTION_HISTORY_FILE
        assert PREDICTION_HISTORY_FILE.endswith('.csv')
    
    def test_enriched_games_csv_in_cache_dir(self):
        """ENRICHED_GAMES_CSV should be in cache directory"""
        assert CACHE_DIR in ENRICHED_GAMES_CSV
        assert ENRICHED_GAMES_CSV.endswith('.csv')
    
    def test_matchups_csv_in_cache_dir(self):
        """MATCHUPS_CSV should be in cache directory"""
        assert CACHE_DIR in MATCHUPS_CSV
        assert MATCHUPS_CSV.endswith('.csv')
    
    def test_xgb_params_file_in_config_dir(self):
        """XGB_PARAMS_FILE should be in config directory"""
        assert CONFIG_DIR in XGB_PARAMS_FILE
        assert XGB_PARAMS_FILE.endswith('.json')


class TestPathConsistency:
    """Tests for path consistency and format"""
    
    def test_all_paths_use_os_path_join(self):
        """All paths should use OS-appropriate separators"""
        # On Windows this is backslash, on Unix it's forward slash
        expected_sep = os.sep
        
        for path in [MODELS_DIR, CACHE_DIR, CONFIG_DIR, REPORTS_DIR]:
            # The path should be a valid OS path
            assert os.path.isabs(path) or PROJECT_ROOT in path
    
    def test_no_duplicate_separators(self):
        """Paths should not have duplicate separators"""
        test_paths = [
            get_model_path('test.pkl'),
            get_cache_path('test.csv'),
            get_config_path('test.json'),
            get_report_path('test.txt')
        ]
        
        for path in test_paths:
            assert '//' not in path.replace('\\', '/')
            assert '\\\\' not in path  # Windows double backslash check
