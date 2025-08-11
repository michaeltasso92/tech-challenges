# AI Microservices Test Suite

This directory contains comprehensive unit tests for all AI microservices in the recommendation system.

## Test Structure

Each microservice has its own test suite located in `services/{service_name}/tests/`:

- **API Service** (`services/api/tests/`): Tests for FastAPI endpoints and recommender logic
- **Parser Service** (`services/parser/tests/`): Tests for JSON guideline parsing
- **Trainer Service** (`services/trainer/tests/`): Tests for model training and scoring
- **Trainer-BiEnc Service** (`services/trainer-bienc/tests/`): Tests for bi-encoder model training
- **UI Service** (`services/ui/tests/`): Tests for Streamlit UI components

## Running Tests

### Run All Tests
```bash
cd AI
python run_tests.py
```

### Run Tests for Specific Service
```bash
python run_tests.py --service services/api
python run_tests.py --service services/parser
python run_tests.py --service services/trainer
python run_tests.py --service services/trainer-bienc
python run_tests.py --service services/ui
```

### Run Tests with Verbose Output
```bash
python run_tests.py --verbose
```

### Install Test Dependencies
```bash
python run_tests.py --install-deps
```

### Run Individual Test Files
```bash
cd services/api
pytest tests/test_api.py -v
```

## Test Coverage

### API Service Tests
- **Health endpoint**: Basic health check functionality
- **Items endpoint**: Retrieval of available items for dropdown
- **Name endpoints**: Individual and batch name lookups
- **Recommendation endpoint**: Core recommendation functionality
- **Error handling**: Network errors, missing data, etc.
- **Recommender class**: Model initialization, name lookup, neighbor enrichment
- **Utility functions**: Artifact loading and error handling

### Parser Service Tests
- **Helper functions**: Node type detection, facing expansion
- **Sequence iteration**: Shelf and bay traversal logic
- **File parsing**: JSON guideline processing
- **Edge cases**: Invalid JSON, missing IDs, single items
- **Integration tests**: Complete parsing workflows

### Trainer Service Tests
- **Scoring algorithm**: PMI-based confidence calculation
- **Data processing**: Left/right neighbor grouping
- **Training workflow**: End-to-end model training
- **Fallback generation**: Global recommendation creation
- **Edge cases**: Empty data, zero counts, missing names

### Trainer-BiEnc Service Tests
- **Text processing**: Row text generation and text map building
- **Data mining**: Pair extraction from parsed data
- **Model training**: Sentence transformer training with examples
- **Encoding**: Embedding generation for all items
- **Artifact saving**: Model, embeddings, and FAISS index saving
- **Data splitting**: Train/validation/test splits by guideline
- **Integration tests**: Complete bi-encoder training workflow

### UI Service Tests
- **Helper functions**: String normalization, badge generation
- **API integration**: Name fetching, recommendation retrieval
- **Data processing**: DataFrame creation and enrichment
- **Search functionality**: Product name search and filtering
- **Error handling**: Network failures, missing data

## Test Dependencies

Each service has its own `requirements.txt` file that includes both runtime and testing dependencies:

- **pytest**: Test framework
- **pytest-mock**: Mocking utilities
- **httpx**: HTTP client for API testing
- **pandas**: Data manipulation for trainer tests
- **numpy**: Numerical operations for scoring tests

## Test Patterns

### Mocking
Tests use extensive mocking to isolate units and avoid external dependencies:
- API calls are mocked to avoid network requests
- File I/O is mocked for faster tests
- External services are mocked for predictable behavior

### Fixtures
Common test data is shared using pytest fixtures:
- Sample guideline data for parser tests
- Mock API responses for UI tests
- Training data for model tests

### Error Handling
Tests cover various error scenarios:
- Network failures
- Invalid data formats
- Missing files
- Permission errors

### Integration Tests
Each service includes integration tests that verify:
- End-to-end workflows
- Data flow between components
- Real file processing (using temporary files)

## Adding New Tests

### For New Endpoints
1. Add test class in appropriate test file
2. Mock external dependencies
3. Test success and failure cases
4. Verify response format and status codes

### For New Functions
1. Create test function with descriptive name
2. Test edge cases and error conditions
3. Use fixtures for common test data
4. Mock external dependencies

### For New Services
1. Create `tests/` directory in service folder
2. Add `__init__.py` file
3. Create test files following naming convention
4. Add service to `run_tests.py` services list

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:
- Fast execution (under 30 seconds for all tests)
- No external dependencies
- Clear pass/fail reporting
- Exit codes for automation

## Best Practices

1. **Isolation**: Each test should be independent
2. **Descriptive names**: Test names should explain what they test
3. **Mocking**: Mock external dependencies
4. **Edge cases**: Test error conditions and boundary cases
5. **Documentation**: Include docstrings explaining test purpose
6. **Maintenance**: Keep tests up to date with code changes
