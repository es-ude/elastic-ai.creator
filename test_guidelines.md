# Test style Guidelines

Files containing tests for a python module should be located in a test directory for the sake of separation of concerns. 
Each file in the test directory should contain tests for one and only one class/function defined in the module. 
Files containing tests should be named according to the rubric
`test_ClassName.py`.
Next, if needed for more specific tests define a class which is a subclass of unittest.TestCase like [test_brevitas_model_comparison](elasticai/creator/translator/brevitas/integrationTests/test_brevitas_model_comparison.py) in the integration tests folder. 
Then subclass it, in this class define a setUp method (and possibly tearDown) to create the global environment. 
It avoids introducing the category of bugs associated with copying and pasting code for reuse. 
This class should be named similarly to the file name.
There's a category of bugs that appear if  the initialization parameters defined at the top of the test file are directly used: some tests require the initialization parameters to be changed slightly. 
Its possible to define a parameter and have it change in memory as a result of a test. 
Subsequent tests will therefore throw errors.
Each class contains methods that implement a test. 
These methods are named according to the rubric
`test_name_condition`

## Unit tests
In those tests each functionality of each function in the module is tested, it is the entry point  when adding new functions. 
It assures that the function behaves correctly independently of others. 
Each test has to be fast, so use of heavier libraries is discouraged.
The input used is the minimal one needed to obtain a reproducible output. 
Dependencies should be replaced with mocks as needed. 

## Integration Tests
Here the functions' behaviour with other modules is tested. 
In this repository each integration function is in the correspondent folder.
Then the integration with a single class of the target, or the minimum amount of classes for a functionality, is tested in each separated file.

## System tests
Those tests will use every component of the system, comprising multiple classes.
Those tests include expected use cases and unexpected or stress tests.

## Adding new functionalities and tests required
When adding new functions to an existing module, add unit tests in the correspondent file in the same order of the module, if a new module is created a new file should be created.
When a bug is solved created the respective regression test to ensure that it will not return.
Proceed similarly with integration tests. 
Creating a new file if a functionality completely different from the others is created e.g. support for a new layer.
System tests are added if support for a new library is added.

## Updating tests
If new functionalities are changed or removed the tests are expected to reflect that, generally the ordering is unit tests -> integration tests-> system tests.
Also, unit tests that change the dependencies should be checked, since this system is fairly small the internal dependencies are not always mocked.

references: https://jrsmith3.github.io/python-testing-style-guidelines.html