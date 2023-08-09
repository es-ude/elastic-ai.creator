# Changelog

<!--next-version-placeholder-->

## v0.54.0 (2023-08-09)

### Feature

* Use binary values instead of hex values to fill the rom template ([`af56c02`](https://github.com/es-ude/elastic-ai.creator/commit/af56c02da42433c2db1a9a2a6ddb3705d213d765))

### Fix

* Use same bit width for all rom values ([`cd609e6`](https://github.com/es-ude/elastic-ai.creator/commit/cd609e65f306e62110fbdc4113f4bb330f960f19))

## v0.53.0 (2023-08-02)

### Feature

* Implement fixed point one dimensional convolution ([`2ea9389`](https://github.com/es-ude/elastic-ai.creator/commit/2ea9389a37eac7be62e26a9727b8824b47fc2085))

### Fix

* Fix missing parameter in tests for conv1d ([`d8f8d4c`](https://github.com/es-ude/elastic-ai.creator/commit/d8f8d4c40ec1576c5dc58a38b2b80d9d4130b4fd))

## v0.52.0 (2023-08-02)

### Feature

* Added Swish activation function precomputed ([`26d292e`](https://github.com/es-ude/elastic-ai.creator/commit/26d292e83183ac0b6bee7afa70f3d616e42b2438))
* Added Swish activation function precomputed ([`fd487b5`](https://github.com/es-ude/elastic-ai.creator/commit/fd487b57bb7d3e7525f935f1533f815e58f1dc0d))
* Added nn module for Swish activation function ([`b7579c9`](https://github.com/es-ude/elastic-ai.creator/commit/b7579c9f1111521064a7fd0366647da7a45e2d7a))
* Implemeted base module for SiLU aka Swish activation function ([`93b5954`](https://github.com/es-ude/elastic-ai.creator/commit/93b59544c1d164de2a9f9362f0aefe1aaae8d7d8))

## v0.51.0 (2023-07-28)

### Feature

* Revert changes and explicitly set semantic release version to v7 instead of v8 ([`2ecf0db`](https://github.com/es-ude/elastic-ai.creator/commit/2ecf0db3c22ce034c4f36a26c96027f0229a4bf0))
* Increase semantic release version to v8.0.4 ([`bb29612`](https://github.com/es-ude/elastic-ai.creator/commit/bb2961243f20ade3e7c4a142601f58fca6e9b5ad))
* Apply proposed semantic release migration procedure ([`d5ea981`](https://github.com/es-ude/elastic-ai.creator/commit/d5ea981cd8852e5790c77d9667187168c34c81e3))
* Seperate semantic release run into multiple steps ([`475b425`](https://github.com/es-ude/elastic-ai.creator/commit/475b425910c4a124c34ca9a68fd5c49b4789541b))
* Enable debug messages ([`36ca597`](https://github.com/es-ude/elastic-ai.creator/commit/36ca597ded38bb3c5343e872ed7cf9cb09065a6f))
* Add debug messages ([`ee09864`](https://github.com/es-ude/elastic-ai.creator/commit/ee09864686d87471617aae4ae65118096d31a6ff))
* Rename custom float to float to improve readability ([`794bfe7`](https://github.com/es-ude/elastic-ai.creator/commit/794bfe79a6a821050b33cc246e9e1cad09e7e682))

### Fix

* Try to fix semantic release ([`0eab187`](https://github.com/es-ude/elastic-ai.creator/commit/0eab187389b3d435be473671d4a593ead8586e78))
* Remove not implemented jvp function ([`0ea4834`](https://github.com/es-ude/elastic-ai.creator/commit/0ea48341c02d116dd3ef2a94e0997ce8e0641b60))

## v0.50.0 (2023-07-11)

### Feature

* Implement custom float arithmetics ([`b72713e`](https://github.com/es-ude/elastic-ai.creator/commit/b72713e3db1e15957e865ed95216a2f180523114))
* Implement RoundToCustomFloat autograd function ([`0794a8e`](https://github.com/es-ude/elastic-ai.creator/commit/0794a8e900d6f87edc03dbd71162e7300e13b5ae))

### Fix

* Return wrong number of values in the backward pass ([`6bcfa4e`](https://github.com/es-ude/elastic-ai.creator/commit/6bcfa4eff8d9b7c0c0461a61800ef68ef6b0cb62))

## v0.49.0 (2023-07-01)

### Feature

* Update readme and add small improvements ([`8f2bbd0`](https://github.com/es-ude/elastic-ai.creator/commit/8f2bbd093e18c15421abab20ecb0f9afbc6d12a1))

### Documentation

* Add minimal example that demonstrates the usage of the creator ([`64030f2`](https://github.com/es-ude/elastic-ai.creator/commit/64030f2eb129ff8275022ab0b8bf4945d42626a8))
* Complete table of contents ([`cf0ef63`](https://github.com/es-ude/elastic-ai.creator/commit/cf0ef63eb628521f14406fb7d59cee53c71c8d60))

## v0.48.1 (2023-06-24)

### Fix

* Only create coverage reports in PR ([`1bd728f`](https://github.com/es-ude/elastic-ai.creator/commit/1bd728f4e8edb6595a35dafd71c5d68263a7358f))

## v0.48.0 (2023-06-24)

### Feature

* Only trigger coverage report when pushing to main ([`b4b23c9`](https://github.com/es-ude/elastic-ai.creator/commit/b4b23c988803165895c14a8357427a3069f09233))
* Add coverage workflow to create reports ([`3f6caca`](https://github.com/es-ude/elastic-ai.creator/commit/3f6caca6a626923ec3d8078320fa9b70092495ee))
* Add pytest-cov dependency ([`a737729`](https://github.com/es-ude/elastic-ai.creator/commit/a7377290ffee7359f6f8c0392960d7038fe2a73b))

### Fix

* Use poetry run to run pytest ([`7058e42`](https://github.com/es-ude/elastic-ai.creator/commit/7058e42cc7fa0849841578f2bafd6a3fc6155f2a))

## v0.47.2 (2023-06-23)

### Fix

* Fix error when passing a cuda tensor to the IdentityStepFunction ([`7f49617`](https://github.com/es-ude/elastic-ai.creator/commit/7f496171a547bae17c69976c35d437428022447f))

## v0.47.1 (2023-06-16)

### Fix

* Remove wrongly committed files ([`4fdea0c`](https://github.com/es-ude/elastic-ai.creator/commit/4fdea0c9ff2db5e8af3f208bbd83d995332d5b85))

## v0.47.0 (2023-06-16)

### Feature

* Simplify project structure ([`81cbcb3`](https://github.com/es-ude/elastic-ai.creator/commit/81cbcb343b26473290609c7715051059127a924b))

## v0.46.1 (2023-06-13)

### Fix

* Fix wrong port definitions ([`9a4c8af`](https://github.com/es-ude/elastic-ai.creator/commit/9a4c8af6f8f8be2bf6fff49c25fc0ca12cbea45a))
* Fix some syntax errors ([`3997bbd`](https://github.com/es-ude/elastic-ai.creator/commit/3997bbdb134a94defd4e32ad1a2eb3aa236d6b96))

## v0.46.0 (2023-06-13)

### Feature

* Use conv1d arithmetics function to implement conv1d module ([`69778be`](https://github.com/es-ude/elastic-ai.creator/commit/69778be7fd1becab2ad5099ebb8d64d4a0db0de5))
* Add conv1d function to arithmetics ([`1cab190`](https://github.com/es-ude/elastic-ai.creator/commit/1cab1901e324eb100f1cbccf6d54fae429210b33))
* Test that conv1d uses different arithmetics ([`7eb01db`](https://github.com/es-ude/elastic-ai.creator/commit/7eb01dbaa2afbbb02410e6fc6272ba02fec7878a))
* Add the ability to sum over dimension ([`c45c0e6`](https://github.com/es-ude/elastic-ai.creator/commit/c45c0e676e1df70bf99c4c943874168781ef2a93))

### Fix

* Quantize weights before inference ([`61153e6`](https://github.com/es-ude/elastic-ai.creator/commit/61153e60d6854bacf0bd2501d96efc3f6e62714e))

## v0.45.0 (2023-06-10)

### Feature

* Simplify usage for the elasticai.creator.nn.vhdl package by adding layers to __init__ ([`2c7c968`](https://github.com/es-ude/elastic-ai.creator/commit/2c7c96858ec9d935389a960baee46e8c506f9b5c))

### Fix

* Fix broken import in base template generator and move it with its template to own folder ([`9eb1f70`](https://github.com/es-ude/elastic-ai.creator/commit/9eb1f70cff10e075712d5bf7e3fc9fcfed2aae19))

## v0.44.0 (2023-06-09)

### Feature

* Check for autowiring protocol violation ([`3f17e00`](https://github.com/es-ude/elastic-ai.creator/commit/3f17e002e050dc92516e4ff5468041f06ebd6760))
* Add AutoWirer ([`f4159c8`](https://github.com/es-ude/elastic-ai.creator/commit/f4159c800fe54cc0fe73fbebdf2ac0410ddac635))
* Add intermediate symbols to rule definitions ([`624b310`](https://github.com/es-ude/elastic-ai.creator/commit/624b310fc9beb130902fdf3269e3f30714fe0c3f))
* Support parsing partial files ([`f2c2eb6`](https://github.com/es-ude/elastic-ai.creator/commit/f2c2eb69ceb8a0b02c1c4617511ccb1528931e23))
* Support parsing partial files ([`8170012`](https://github.com/es-ude/elastic-ai.creator/commit/817001208b774e57cfb27fb4d4ee9d704541c9f8))
* Add standalone parser module ([`5a9b141`](https://github.com/es-ude/elastic-ai.creator/commit/5a9b141285fefecf61f581417061428cda382ad5))
* Add basic vhdl parsing ([`5df2a3f`](https://github.com/es-ude/elastic-ai.creator/commit/5df2a3ff4e9ba7ec33398a267cd983ad886d1fe7))
* Port expansion/template based on autowiring protocol ([`0d14618`](https://github.com/es-ude/elastic-ai.creator/commit/0d146181c8b789b09871af43654ca2d83ea55ddb))
* **template:** Make precomputed scalar functions bufferless ([`89986fa`](https://github.com/es-ude/elastic-ai.creator/commit/89986fad041c89d0543fe9a22946e5f5f49e2b61))

### Fix

* Use new Sequential constructor ([`6bb111b`](https://github.com/es-ude/elastic-ai.creator/commit/6bb111b748567502c23a48a52d7e477645969996))
* Port def and impl of monotonous function design ([`2d423d4`](https://github.com/es-ude/elastic-ai.creator/commit/2d423d46faa86fbf43cb8ba1d01aafe92c5bfa23))
* Children of sequential layer determine signal widths ([`3dd5c0c`](https://github.com/es-ude/elastic-ai.creator/commit/3dd5c0cc4f7a52c7b3a86cec437005b86aa0a509))
* Remove obsolete parsing functionality ([`7f85d05`](https://github.com/es-ude/elastic-ai.creator/commit/7f85d05aa3da2e0fd7c266bfc9c1aad573adecc4))
* Adjust tests to follow previous change ([`c328bd5`](https://github.com/es-ude/elastic-ai.creator/commit/c328bd565d6ba84a9d1fab788051c3e884ea2094))
* Correct tuple type annotation ([`f0e7da0`](https://github.com/es-ude/elastic-ai.creator/commit/f0e7da0cf186015004970102f2b9b57a9f839585))

## v0.43.0 (2023-06-09)

### Feature

* Add tests for the FPMonotonouslyIncreasingModule ([`9ba64ae`](https://github.com/es-ude/elastic-ai.creator/commit/9ba64ae3d253db76a6368c5e561ce28bcec2aab5))
* Introduce FPMonotonouslyIncreasingModule to easily add new activations ([`b78c922`](https://github.com/es-ude/elastic-ai.creator/commit/b78c9225f7f70ec329bee5705c11d9e7b1392c41))

### Fix

* Set correct signal names for x and y address ([`5354a2a`](https://github.com/es-ude/elastic-ai.creator/commit/5354a2a0e85bc0788f5d74377c1a685e9d0e0de7))
* Use elsif in lookup table ([`f375ba3`](https://github.com/es-ude/elastic-ai.creator/commit/f375ba3784bf92887e689f77f592dfc2fa2c7e2c))
* Increase default sampling intervall ([`07620d3`](https://github.com/es-ude/elastic-ai.creator/commit/07620d3e2ee9db1bc6aa081a15274cb79b5ee4b0))

## v0.42.0 (2023-06-08)

### Feature

* Reimplement hard tanh activation function ([`9b86f9d`](https://github.com/es-ude/elastic-ai.creator/commit/9b86f9d440cc991d624a6f3492a3caf7419bdbf3))
* Add working hardsigmoid implementation ([`db03ff0`](https://github.com/es-ude/elastic-ai.creator/commit/db03ff080f878c9b9fe54303ead97c673022f3a1))
* Make sure that inplace parameter is fixed defined ([`79b7a1e`](https://github.com/es-ude/elastic-ai.creator/commit/79b7a1eea0cb71f5a838cfebf02970927410f594))

## v0.41.0 (2023-06-08)

### Feature

* Add fixed point ReLU module ([`62c1555`](https://github.com/es-ude/elastic-ai.creator/commit/62c15557fc515c89644c674aef9fc39d22ab672f))

## v0.40.0 (2023-06-04)

### Feature

* Simplify the use of the sequential layer (same as in torch) ([`9fad15d`](https://github.com/es-ude/elastic-ai.creator/commit/9fad15d774f3573fb26f168295f9bd2ae5cdd046))
* Improve performance of the identity step autograd function ([`46f036c`](https://github.com/es-ude/elastic-ai.creator/commit/46f036c8fb2d007d21e32214ac92d4d9aa2fe9d1))
* Add quantized tanh implementation with lookup tables ([`3a1fb10`](https://github.com/es-ude/elastic-ai.creator/commit/3a1fb10944e566ca33e3e745b939b6700421fdb9))
* Implement bufferless component interface for precomputed scalar function ([`f701a57`](https://github.com/es-ude/elastic-ai.creator/commit/f701a57db54e0d5f3e5e43047725b28646cb5f15))
* Pass step lut to identity step function and improve readablility ([`c1b6747`](https://github.com/es-ude/elastic-ai.creator/commit/c1b67473c33ddc27590068472dcff6969f9e7135))
* Rename autograd function and pass step lut to autograd function ([`d607e98`](https://github.com/es-ude/elastic-ai.creator/commit/d607e98bd14dfa1ae23e9726b2046baaede21361))
* Implement autograd fn to map inputs to a subset of inputs ([`26c6ec7`](https://github.com/es-ude/elastic-ai.creator/commit/26c6ec7a203eea4fed4c3eb3d5c3e4893acb545f))
* Add a function to easily compare tensors with pytest ([`24e737e`](https://github.com/es-ude/elastic-ai.creator/commit/24e737eaea48044df3e8addaca0d1cc804a3b6f4))
* Add experimental precomputed tanh in fixed point ([`0e76d03`](https://github.com/es-ude/elastic-ai.creator/commit/0e76d03b6d0f23d8932b94bb7728cbeea2de0289))

### Fix

* Fix that last io pair was dropped when calling save_to function ([`2bc46ac`](https://github.com/es-ude/elastic-ai.creator/commit/2bc46ac9c535b65ef7a3dc5cbe12b27d253c3b37))
* Fix missing creation of a subpath in the save_to function ([`2a4dbdf`](https://github.com/es-ude/elastic-ai.creator/commit/2a4dbdf2f6fce4de567281002dd4640ff3ae54ed))

## v0.39.0 (2023-05-19)
### Feature
* Implement batch normed linear layer ([`9322f6f`](https://github.com/es-ude/elastic-ai.creator/commit/9322f6f699f9884273c3f9815b9a026c9f7840ae))

### Fix
* Allow to set affine and bias equals false in translate function ([`b351284`](https://github.com/es-ude/elastic-ai.creator/commit/b351284335a77caec838a8f4ea57684e429cc35b))
* Remove dequantize ([`c111022`](https://github.com/es-ude/elastic-ai.creator/commit/c111022854ce6965b705b3a3de296e032d7ff107))

## v0.38.0 (2023-05-09)
### Feature
* Add check that all variables are filled when saving a template ([`c988d2b`](https://github.com/es-ude/elastic-ai.creator/commit/c988d2bc203790ba8ab900e8a2de6996b22d6fcb))
* Add function to get all unfilled variables of a template ([`d635cb6`](https://github.com/es-ude/elastic-ai.creator/commit/d635cb6098735b451aea259a8a6f15619bfcd64f))
* Write function of InMemoryFile and OnDiskFile now takes Template object ([`a867ea1`](https://github.com/es-ude/elastic-ai.creator/commit/a867ea15980b8ca1390327f2999c4d7b91ef3041))

### Fix
* Remove broken lstm implementation ([`c524ca2`](https://github.com/es-ude/elastic-ai.creator/commit/c524ca20cc49333007c4e0bbfa167912580e5c01))
* Add variable ([`229d452`](https://github.com/es-ude/elastic-ai.creator/commit/229d452d0c2f798ee1dd0124f50be8f01d69ede4))
* Fix not inserted process name ([`dbabea0`](https://github.com/es-ude/elastic-ai.creator/commit/dbabea07c888a5309d9ca55cd2c01ae0debea57d))

## v0.37.2 (2023-05-07)
### Fix
* Try manual publishing ([`c8b6c35`](https://github.com/es-ude/elastic-ai.creator/commit/c8b6c355896c1f3b0630c227af8414f281b5d3ff))

## v0.37.1 (2023-05-07)
### Fix
* Try to fix semantic release ([`2625e89`](https://github.com/es-ude/elastic-ai.creator/commit/2625e8982c021cbf5b778e95194cc53170ab0afb))

## v0.37.0 (2023-05-05)
### Feature
* Assert that all inserted variables exists in template and remove AbstractBaseTemplate ([`51f1a08`](https://github.com/es-ude/elastic-ai.creator/commit/51f1a0883a8d0a54caee66080ef85f84049ad806))

## v0.36.0 (2023-04-26)
### Feature
* Autogenerate sequential signal connections ([`6dfca07`](https://github.com/es-ude/elastic-ai.creator/commit/6dfca078b735a3387b65c20de601426ea27371c6))
* **unit:** Add tests for sequential model with two layer ([`df73a4f`](https://github.com/es-ude/elastic-ai.creator/commit/df73a4fb27a8867a4b633c4ffdd737ead34d2f16))
* Test signal definitions, layer connections and instantiations separately ([`65201c8`](https://github.com/es-ude/elastic-ai.creator/commit/65201c83bae07c62efcd705f67f34d9ff88da557))
* **unit:** Test all subdesigns generated by sequential layer gets a unique name ([`009405b`](https://github.com/es-ude/elastic-ai.creator/commit/009405bc64cd5e8a86909330bb450ee58ee98289))
* Sequential layer can have a name ([`9e46938`](https://github.com/es-ude/elastic-ai.creator/commit/9e46938e9e5fc6960e70bef26aa72ec51566a007))
* Introduce abstract Translatable class ([`5d9fa2d`](https://github.com/es-ude/elastic-ai.creator/commit/5d9fa2d167a8c46c301bb4a0da25718b1fcf0dee))
* Add indentations to template ([`aa254d1`](https://github.com/es-ude/elastic-ai.creator/commit/aa254d12f38712e798db9b31a5a58e197a44121a))
* Implement translatable identity module ([`54327fa`](https://github.com/es-ude/elastic-ai.creator/commit/54327fa3e45ca3617d642134ca8d842e7d2afc4c))
* Add translate_to_vhdl function ([`ba0edc2`](https://github.com/es-ude/elastic-ai.creator/commit/ba0edc25b93075cbb2d104c2216dcc15df36c13c))

### Fix
* Correct expected connections ([`2fb0f8e`](https://github.com/es-ude/elastic-ai.creator/commit/2fb0f8edc45a7a38e2a9b7433dee90f139b10006))
* **unit:** Fix that test ignores parameter ([`f448919`](https://github.com/es-ude/elastic-ai.creator/commit/f448919ff4882696c0991d6aec3608616e258596))
* Fix syntax error ([`396f5c4`](https://github.com/es-ude/elastic-ai.creator/commit/396f5c45b382454d6cc97e4be573fcfe45a4592a))
* Fix syntax errors ([`f9b57e4`](https://github.com/es-ude/elastic-ai.creator/commit/f9b57e4f8173dc0bd52c21b1da351304ceb5a122))
* Add missing save_to function ([`ef24ee2`](https://github.com/es-ude/elastic-ai.creator/commit/ef24ee21672099359867bc4a74f5804af0c10158))
* **unit:** Fix tests and remove hard sigmoid test in sequential test case ([`a1ada6f`](https://github.com/es-ude/elastic-ai.creator/commit/a1ada6f0ceec750bb80abf866d28f96719f2f1f9))
* Set correct resource options in rom and fix signal definitions ([`2c2964c`](https://github.com/es-ude/elastic-ai.creator/commit/2c2964ceaa746163ebbeaef09181e09c06ecb4f2))

## v0.35.0 (2023-04-17)
### Feature
* Use fixed base template ([`432dfd9`](https://github.com/es-ude/elastic-ai.creator/commit/432dfd9518a0a33a7ba08cf95436f9472b274b52))
* Generate template from manifest.toml ([`51276a0`](https://github.com/es-ude/elastic-ai.creator/commit/51276a01de5ff37bedc598f5c758e3dc681aa49c))
* Generate first base template ([`a65d72e`](https://github.com/es-ude/elastic-ai.creator/commit/a65d72ea1ad2dd87a0443b56711d11ce321d14b6))

## v0.34.0 (2023-04-06)
### Feature
* Make precomputed scalar functions use unified interface ([`6b59da5`](https://github.com/es-ude/elastic-ai.creator/commit/6b59da53a896db7676119de2f74129bcc47287ed))
* Binary_arithmetics ([`54e38d5`](https://github.com/es-ude/elastic-ai.creator/commit/54e38d57f27db2d8d0baff5fee3c35a91e26ecd9))

### Fix
* Correct import paths ([`169f868`](https://github.com/es-ude/elastic-ai.creator/commit/169f8686108845702f01482170df53e3fabbfe8b))

## v0.33.3 (2023-04-06)
### Fix
* Remove DualPort2ClockRam design ([`f9224c6`](https://github.com/es-ude/elastic-ai.creator/commit/f9224c6809b3a6f72bfe0405419de494b099b17c))
* Set correct rom names ([`9570826`](https://github.com/es-ude/elastic-ai.creator/commit/95708269900ca99b79da9ba37078f593724e5d17))

### Documentation
* Remove deprecated documentation ([`11b9945`](https://github.com/es-ude/elastic-ai.creator/commit/11b9945bf3b6bf96899a09751963a93eb98d846d))

## v0.33.2 (2023-03-23)
### Fix
* Small import fix ([`07d2e29`](https://github.com/es-ude/elastic-ai.creator/commit/07d2e29c36e60d35066d2145782223aa42d64519))
* **translation:** Add missing rom files and calculate correct twos complement ([`f700409`](https://github.com/es-ude/elastic-ai.creator/commit/f70040956b7637844a471a5eff171d9cc6ba4c72))
* **translation:** Add missing ROMs and set correct names in fp_linear1d template ([`ad4c6f0`](https://github.com/es-ude/elastic-ai.creator/commit/ad4c6f095102965ff1dffa83dab4f2cb9749ce49))
* Fix type annotation ([`8da1107`](https://github.com/es-ude/elastic-ai.creator/commit/8da1107b2640d695816c71dd3980c0783b522122))
* **unit:** Fix failing unittests that are using the linear1d layer and design ([`ff582e1`](https://github.com/es-ude/elastic-ai.creator/commit/ff582e185ea01cc6282cb4553e14701e88a9d8f8))

## v0.33.1 (2023-03-15)
### Fix
* Usage of lstm output in lstm_network impl ([`2e16141`](https://github.com/es-ude/elastic-ai.creator/commit/2e1614184cdaa073fdcc686b891748861fe5c7cc))
* Wrong fixed point config object used for linear layers ([`3626113`](https://github.com/es-ude/elastic-ai.creator/commit/36261136add4b4d378598dc8c9e858240f6557c5))

## v0.33.0 (2023-03-15)
### Feature
* Add rom design for saving weights ([`75862b7`](https://github.com/es-ude/elastic-ai.creator/commit/75862b7db4e64173daf7e6cdcb8413b0f510d396))

### Fix
* Correctly pad rom memory ([`fe768d5`](https://github.com/es-ude/elastic-ai.creator/commit/fe768d5f93c34ade65c24479c70f3528c66b0408))

## v0.32.1 (2023-03-14)
### Fix
* Typo in test for lstm cell designs ([`2ffeaec`](https://github.com/es-ude/elastic-ai.creator/commit/2ffeaecf3ba7c3c0946c57ab3bee92af55746887))
* Set library for lstm_cell ([`2b3a565`](https://github.com/es-ude/elastic-ai.creator/commit/2b3a565039672ca89a1c5f593db5a5f32742f771))

## v0.32.0 (2023-03-14)
### Feature
* Add linear layer to lstm network ([`bccb50c`](https://github.com/es-ude/elastic-ai.creator/commit/bccb50cd6e3bc4e3e3115a41e051a1b962f6be52))
* Add linear layer to lstm network ([`48982f0`](https://github.com/es-ude/elastic-ai.creator/commit/48982f0aca675098b77edb2c8419b09ebc388835))
* **translation:** Add support for single buffered module to sequential ([`5402782`](https://github.com/es-ude/elastic-ai.creator/commit/5402782c0c37a6838b77b19d8040d256217d72ba))
* **translation:** Sequential layer with bufferless layers ([`d7cea69`](https://github.com/es-ude/elastic-ai.creator/commit/d7cea69ad0696f63e00762991e7407ad09d8a94c))
* Add data flow node, sink node and source node ([`9a511de`](https://github.com/es-ude/elastic-ai.creator/commit/9a511de4d2618c3131abcd3c481b918ffa96545e))
* Introduce vhdl_design class ([`2431ba4`](https://github.com/es-ude/elastic-ai.creator/commit/2431ba40b71c19dff161ea9b78d7b5277970a6f9))
* Add logic and logic vector signals ([`551f241`](https://github.com/es-ude/elastic-ai.creator/commit/551f24113be03b45ab1811cb734e521671620d89))
* Add connectable base in/out signals ([`fea05ed`](https://github.com/es-ude/elastic-ai.creator/commit/fea05ed0507550c23701bf6f5e3a562b68af73d4))
* Introduce vhdl_design class ([`20566f6`](https://github.com/es-ude/elastic-ai.creator/commit/20566f600383ccb68fed60483bede9db5436913f))
* Add logic and logic vector signals ([`1947baa`](https://github.com/es-ude/elastic-ai.creator/commit/1947baac032e1b3958344779a00b84615b5581a1))
* Add connectable base in/out signals ([`7ad67f9`](https://github.com/es-ude/elastic-ai.creator/commit/7ad67f916815b692daddae98d4c93b9a5eb21641))

### Fix
* **translation:** Correct values for x/y_address_width ([`c7af1af`](https://github.com/es-ude/elastic-ai.creator/commit/c7af1af71ef9319ed2ee7fffd7afcbaa5ffda580))
* Tests and remove type annotations leading to deps ([`75ed6cc`](https://github.com/es-ude/elastic-ai.creator/commit/75ed6cc4f3a92b80656433b8209c0c932595900e))
* Typing ([`b0bfa39`](https://github.com/es-ude/elastic-ai.creator/commit/b0bfa39b98555b37f0d2626a235ac74987e2c9ad))
* Fix unit tests after major rebase ([`3b596e9`](https://github.com/es-ude/elastic-ai.creator/commit/3b596e9c20e302bbf42efda7577e01498c05bc6c))
* Type annotations for tracing module ([`da598a9`](https://github.com/es-ude/elastic-ai.creator/commit/da598a92fc8f76b3c19d0b960d77122b82d171ac))
* Fix incorrect vector signal initialization ([`3b23f7a`](https://github.com/es-ude/elastic-ai.creator/commit/3b23f7a64afda8cd2ee320f6af7dc372f9daf5e2))
* Fix incorrect vector signal initialization ([`3c68255`](https://github.com/es-ude/elastic-ai.creator/commit/3c68255057dad325ab4ba89601f6f1e2384f0d95))

## v0.31.0 (2023-02-22)
### Feature
* **translation:** Add missing suffixes ([`cb05d0f`](https://github.com/es-ude/elastic-ai.creator/commit/cb05d0f3f8665ac98c0cff70cbb2dbd8d2a5b2f2))

## v0.30.4 (2023-02-16)
### Fix
* **translation:** Get rid of the duplicated suffix on rom component ([`9cd0e0b`](https://github.com/es-ude/elastic-ai.creator/commit/9cd0e0be9481a286820eea5c8d5bdc9d28fcc0d8))

## v0.30.3 (2023-02-16)
### Fix
* **unit:** Add rounding to prevent tests from failing due to floating point loss ([`b7314b7`](https://github.com/es-ude/elastic-ai.creator/commit/b7314b797ef39c2f693554821ec7bb3d96689661))
* **template:** Linear layer template ([`96bdf03`](https://github.com/es-ude/elastic-ai.creator/commit/96bdf030ca4c27d67a4978e3b8609ef57c40a01e))

## v0.30.2 (2023-02-15)
### Fix
* Use non-static path to example folder ([`613a152`](https://github.com/es-ude/elastic-ai.creator/commit/613a152e65fbe0f7116a1f772fea8a3836d888af))
* Ignore single import mypy error ([`dd85159`](https://github.com/es-ude/elastic-ai.creator/commit/dd851590719ec76ab66dc9d908493991fc235e7e))

## v0.30.1 (2023-02-04)
### Fix
* **unit:** Make test more deterministic ([`97fd410`](https://github.com/es-ude/elastic-ai.creator/commit/97fd4101af93cf17d446cb0cb38a419080d5bee6))

## v0.30.0 (2023-02-04)
### Feature
* Small example for translating combination of lstm and linear layer ([`12e7101`](https://github.com/es-ude/elastic-ai.creator/commit/12e7101e8c62e8424bc2ed580cfbe645e8d33510))
* **translation:** Integrate hard tanh layer ([`eb74d3a`](https://github.com/es-ude/elastic-ai.creator/commit/eb74d3a3671616db37ba8f554332ca1ddc33dffe))
* **translation:** Lstm uses fp hard sigmoid ([`fd265ac`](https://github.com/es-ude/elastic-ai.creator/commit/fd265ac3e1ef7f11e28236705e4a38760462bddc))
* Add example to demonstrate that the new kinds of layers are trainable ([`231e325`](https://github.com/es-ude/elastic-ai.creator/commit/231e325815c469596c63259c5f345dc9afb0f3b7))
* **nn:** Remove quantized_forward function and adopt tests ([`c865c73`](https://github.com/es-ude/elastic-ai.creator/commit/c865c73a53e89c40ecebc9c4b49ba6d5c14256c1))
* **nn:** Implement concept of arithmetics ([`e7ad504`](https://github.com/es-ude/elastic-ai.creator/commit/e7ad50471e2ac7300e0db781bd37cbba1364a5e6))
* **nn:** Remove input_quant and param_quant and add quantize function to arithmetics ([`ee91e42`](https://github.com/es-ude/elastic-ai.creator/commit/ee91e42801b0d1163a0d52130fc578477da60c74))
* **nn:** Integrate arithmetics for the linear layer ([`a961558`](https://github.com/es-ude/elastic-ai.creator/commit/a9615581159ba4b962fac8458d9b76de0a61d98f))
* **nn:** Rename quant_typings module to quantization and implement FakeQuant ([`0e5f24a`](https://github.com/es-ude/elastic-ai.creator/commit/0e5f24aeb9f43258f9e971ffa777c585faff05f0))
* **unit:** Improve TensorTestCase class ([`d4273a6`](https://github.com/es-ude/elastic-ai.creator/commit/d4273a60c169669ddba5f80636d1430b69c77d90))
* **unit:** Add unit tests for the fixed point quant/dequant autograd functions ([`f82431c`](https://github.com/es-ude/elastic-ai.creator/commit/f82431c164b9536899d0cca9b391a057add8187a))
* **unit:** Add unit tests for the LSTMBase layer ([`589f803`](https://github.com/es-ude/elastic-ai.creator/commit/589f803fd858b22985485d795f4441a9abf97742))
* **integration:** Convert example translate_linear_model to automated integration test ([`5d92d0b`](https://github.com/es-ude/elastic-ai.creator/commit/5d92d0b15d8c0a1d76f842fd7a8bbc591bd1cf18))
* Convert example parametrize_convolution to automated integration test ([`3dde1c2`](https://github.com/es-ude/elastic-ai.creator/commit/3dde1c250fa4ebb617bbd543c9b26cb320d430f7))
* **vhdl:** Start implementing lstm base module ([`b154ca5`](https://github.com/es-ude/elastic-ai.creator/commit/b154ca5525c00f735150c21f64324da87328ba5e))
* **vhdl:** Implement quantized forward function of the fixed point lstm cell ([`7818e15`](https://github.com/es-ude/elastic-ai.creator/commit/7818e15bc6c41454090b77fe5df7a8e7930ab570))
* **vhdl:** Implement and test lstm cell base class and start implementing fp lstm cell ([`f458fb6`](https://github.com/es-ude/elastic-ai.creator/commit/f458fb6c216385a119774a3f98788941e13ed5c9))
* **vhdl:** Start implementing lstm base layer ([`39ce891`](https://github.com/es-ude/elastic-ai.creator/commit/39ce891d56be59d5a20a36889b0e9c2f13e00bd1))
* **vhdl:** Implement FixedPointHardTanh layer ([`ed72810`](https://github.com/es-ude/elastic-ai.creator/commit/ed728101fb596a08e1a76d936d04306a066c50b5))

### Fix
* **translation:** Fix errors in the lstm template and remove lstm_common component ([`c4a28ce`](https://github.com/es-ude/elastic-ai.creator/commit/c4a28ce2f40dc84e7a5e4470c62a40911b73901f))
* **translation:** Add layer_name to all vhdl templates and components ([`2d9c47d`](https://github.com/es-ude/elastic-ai.creator/commit/2d9c47dc60642d94efeb58cc3014f6a7790a6f26))
* **translation:** Change not existing layer_id field to layer_name ([`f7425c5`](https://github.com/es-ude/elastic-ai.creator/commit/f7425c515395243962db1517116b9961b1668cd7))
* Fix some mypy errors and remove unused imports ([`08e2362`](https://github.com/es-ude/elastic-ai.creator/commit/08e2362fa32efd13e388140ad58c93b0e79229b3))
* **translation:** Use model.children() instead of model.modules() to avoid recursion ([`a3c349b`](https://github.com/es-ude/elastic-ai.creator/commit/a3c349b13af0fef383b494850973d8ff9ac2dd68))
* **translation:** Remove sigmoid_resolution ([`dd4f033`](https://github.com/es-ude/elastic-ai.creator/commit/dd4f03366920f1a3774772a16a49efaa8756d249))
* **translation:** Rename to .tpl.vhd ([`fe3c85c`](https://github.com/es-ude/elastic-ai.creator/commit/fe3c85cd77d0f2fefb90f2d3ff6eadde8570d000))
* **nn:** Fix LSTMCell raises Error for unbatched input data and add a test for this case ([`5ce3e21`](https://github.com/es-ude/elastic-ai.creator/commit/5ce3e2125b4bcd1115d77ebe5c833e52d58bad77))
* **translation:** Infer fixed_point_factory of linear and lstm in build functions ([`81df686`](https://github.com/es-ude/elastic-ai.creator/commit/81df686fe13db5f85c91b65c73713b7da8e6c64f))
* **translation:** Change torch LSTM layer to our FixedPointLSTM layer ([`5e7a39a`](https://github.com/es-ude/elastic-ai.creator/commit/5e7a39a78684c09a1d374476f8fb611019ae994f))
* **unit:** Remove unused OperationType type and FakeQuant class ([`596dbd8`](https://github.com/es-ude/elastic-ai.creator/commit/596dbd8cdf3cde67eedea2779a35ff682c9ac9f7))
* **unit:** Fix unit and integration tests to use the new layers correctly ([`0553017`](https://github.com/es-ude/elastic-ai.creator/commit/05530178cf7fb64dc88cab82b89c24b2a1406e8d))
* **nn:** Fix imports and use new FixedPointConfig features ([`e8c74c3`](https://github.com/es-ude/elastic-ai.creator/commit/e8c74c34ec1c5a4b5189d74f2a19a993a5ae9779))
* **translation:** Add similar concept of translation arguments to fix the translation process ([`e387ae2`](https://github.com/es-ude/elastic-ai.creator/commit/e387ae26918fbe8e4a0ee01ccc4361849746bd66))
* Adapt basic qtorch example to recent changes of the creator ([`a17d900`](https://github.com/es-ude/elastic-ai.creator/commit/a17d9006240a67da97b8a539620aa1974e07e942))
* **vhdl:** Remove layer_name parameter ([`7a83b1e`](https://github.com/es-ude/elastic-ai.creator/commit/7a83b1eed3095a8b7f90438c78ba24bba6e44958))
* **vhdl:** Fix wrong return type ([`eb53ed9`](https://github.com/es-ude/elastic-ai.creator/commit/eb53ed972ec9078f6c405ecd7c92043eaf8ed419))

### Documentation
* Add commit types and scopes ([`e759fd3`](https://github.com/es-ude/elastic-ai.creator/commit/e759fd38fb41d413ccf03617f84f87f6df9aeb12))

## v0.29.0 (2022-12-16)
### Feature
* Set pypi project api token ([`37ba8c9`](https://github.com/es-ude/elastic-ai.creator/commit/37ba8c9794acc6b4bdf64087c98c61172446fcb6))

## v0.28.0 (2022-12-16)
### Feature
* **qat:** Remove constraints ([`6b7b483`](https://github.com/es-ude/elastic-ai.creator/commit/6b7b4835dc9f9f6b6fc83bc619727aa948c19161))
* **examples:** Update qlstm sine wave example to the correctly implemented QLSTM layer ([`dc62cd2`](https://github.com/es-ude/elastic-ai.creator/commit/dc62cd2aa05067b164009301ab7c5e110797c503))
* **qat:** Add constraint type ([`dc4c4e5`](https://github.com/es-ude/elastic-ai.creator/commit/dc4c4e57a9615a9be6941ecc750d3838458ff919))

### Fix
* **qat:** Fix error when passing flat input data to _QLSTMBase and batch_first set to True ([`29918d1`](https://github.com/es-ude/elastic-ai.creator/commit/29918d11c508e3e91fe00a0e07988be0ed198b35))
* **qat:** Fix the problem of wrong shapes for the QLSTM layer ([`b75f478`](https://github.com/es-ude/elastic-ai.creator/commit/b75f47804016a3dfdad3f8d2dd575f4252cac5ff))
* **qat:** Fix circular dependency ([`1d5615b`](https://github.com/es-ude/elastic-ai.creator/commit/1d5615bf81757bf16904eb75c33fead69a68dd43))

## v0.27.0 (2022-12-15)
### Feature
* **vhdl:** Distinguish x/y width ([`2f52100`](https://github.com/es-ude/elastic-ai.creator/commit/2f52100d32502520ce66a240bae90dd48e070ebd))
* **vhdl:** Introduce HWBlockCollection ([`a80bda2`](https://github.com/es-ude/elastic-ai.creator/commit/a80bda2d705992030b18649ff99f3a6ce75d7ef3))
* **vhdl:** Introduce HWEquivalentGraph ([`844bb84`](https://github.com/es-ude/elastic-ai.creator/commit/844bb84a2d36e50f3de7ae4b713d370011d3240e))
* **vhdl:** Add module_nodes to graph decorator ([`6d0a612`](https://github.com/es-ude/elastic-ai.creator/commit/6d0a61217b36b9db8e9df19210e5f0d3aeed4ef2))
* **vhdl:** Implement HWBlocks interface for sigmoid,linear ([`0177373`](https://github.com/es-ude/elastic-ai.creator/commit/0177373eeddfa9c32100777bbcd7a94765dc1122))
* **vhdl:** Extend code file with parameters ([`4833f8b`](https://github.com/es-ude/elastic-ai.creator/commit/4833f8b2d5553cf02d322b8485587612cd67a9e8))
* **vhdl:** Introduce HWBlocks ([`ab03eaf`](https://github.com/es-ude/elastic-ai.creator/commit/ab03eaf28c74483fcd9dbd78d247d39e248bdea1))
* **vhdl:** Generate layer instantiations ([`7a75fc3`](https://github.com/es-ude/elastic-ai.creator/commit/7a75fc31780a6173424ffdcf3129bc60d5a83e59))
* **vhdl:** Generate vhdl signal definitions ([`c593d3d`](https://github.com/es-ude/elastic-ai.creator/commit/c593d3d501082595d4918be3c3425b6d9c636332))
* **vhdl:** Generate vhdl signal definitions ([`53408f6`](https://github.com/es-ude/elastic-ai.creator/commit/53408f6cb9daa5c44931e880fda0712c2924b822))
* **vhdl:** Tracer records reference to module for call_module nodes ([`20ed7da`](https://github.com/es-ude/elastic-ai.creator/commit/20ed7dab9677e476925a8b1250cbbc2004d43246))
* **vhdl:** Add hw equivalent module tracer ([`3f2c2c7`](https://github.com/es-ude/elastic-ai.creator/commit/3f2c2c7acc5046131d420d513a4bb3d3981ac0c5))
* **vhdl:** Generate portmap output_address ([`c6a26a6`](https://github.com/es-ude/elastic-ai.creator/commit/c6a26a61d98c90fa29b02e6619116e67a4a67ac5))
* **vhdl:** Support generation of layer connections ([`1d43c42`](https://github.com/es-ude/elastic-ai.creator/commit/1d43c4212ef54c5488df7e7dc3829df31a7e8484))
* **vhdl:** Distinguish x/y width ([`73549f9`](https://github.com/es-ude/elastic-ai.creator/commit/73549f94a0c582170e2f43baea4afcb4c9c20124))
* **vhdl:** Introduce HWBlockCollection ([`cdcb324`](https://github.com/es-ude/elastic-ai.creator/commit/cdcb324abe3c69893a782df075b24d734f244a6c))
* **vhdl:** Introduce HWEquivalentGraph ([`f0bdd73`](https://github.com/es-ude/elastic-ai.creator/commit/f0bdd73d6e6e6ed9c8306a7771443e4d13e874ce))
* **vhdl:** Add module_nodes to graph decorator ([`bee0438`](https://github.com/es-ude/elastic-ai.creator/commit/bee0438fb9b35d666998f4f516a1469c729b5829))
* **vhdl:** Implement HWBlocks interface for sigmoid,linear ([`53e05c7`](https://github.com/es-ude/elastic-ai.creator/commit/53e05c7b772f8576b4f221e610360dc52601d852))
* **vhdl:** Extend code file with parameters ([`2bdfca3`](https://github.com/es-ude/elastic-ai.creator/commit/2bdfca352b05756bb911eafb2b702f6536561b26))
* **vhdl:** Introduce HWBlocks ([`141148f`](https://github.com/es-ude/elastic-ai.creator/commit/141148f13c40725755a1b02b24d8899e01ae9ced))
* **vhdl:** Generate layer instantiations ([`925b837`](https://github.com/es-ude/elastic-ai.creator/commit/925b837d33120d4bd1abdd8cae812d89d4979a9a))
* **vhdl:** Generate vhdl signal definitions ([`c76d03d`](https://github.com/es-ude/elastic-ai.creator/commit/c76d03db443cffd831abee60a8546aa3547c5fe6))
* **vhdl:** Generate vhdl signal definitions ([`5da3986`](https://github.com/es-ude/elastic-ai.creator/commit/5da3986472a65e7f15cbedd3cba473ad4d67dde9))
* **vhdl:** Tracer records reference to module for call_module nodes ([`ea1f0ee`](https://github.com/es-ude/elastic-ai.creator/commit/ea1f0ee893c11065bdf17086badd248b998d29de))
* **vhdl:** Add hw equivalent module tracer ([`fcb2e10`](https://github.com/es-ude/elastic-ai.creator/commit/fcb2e102f5409a2e1dc358ce26e4cba6110a7e24))
* **vhdl:** Generate portmap output_address ([`33e66d9`](https://github.com/es-ude/elastic-ai.creator/commit/33e66d99b5b8c0801c93e520463ea92c6392e2b8))
* **vhdl:** Support generation of layer connections ([`fdd3176`](https://github.com/es-ude/elastic-ai.creator/commit/fdd3176ba5d4652718e76dfd74dc92167f86b4f4))

### Fix
* **vhdl:** Remove obsolete vhdl formatter ([`83d81e3`](https://github.com/es-ude/elastic-ai.creator/commit/83d81e348152e047482ccc45a2ccaf6173f772d9))
* **onnx:** Remove unmaintained onnx support ([`dc773d3`](https://github.com/es-ude/elastic-ai.creator/commit/dc773d39fe2c0ea5785e3fb0bf7a43f3bf83495f))
* **vhdl:** Remove obsolete vhdl formatter ([`128ba6b`](https://github.com/es-ude/elastic-ai.creator/commit/128ba6bdbecd8763f77cc6862373446f5418201e))
* **onnx:** Remove unmaintained onnx support ([`c200394`](https://github.com/es-ude/elastic-ai.creator/commit/c200394239ff58ee31e5273d5999d731fbe5daca))

### Documentation
* **readme:** Move tests and remove deprecated lines ([`4a074a8`](https://github.com/es-ude/elastic-ai.creator/commit/4a074a87fb31df535d415c2ab6aede7e4d7d8949))

## v0.26.1 (2022-11-30)
### Fix
* **vhdl:** Remove layer_name parameter ([`1bb40cd`](https://github.com/es-ude/elastic-ai.creator/commit/1bb40cd0e44f7f207f60ffbb33e8c59f00b64e82))

## v0.26.0 (2022-11-23)
### Feature
* **vhdl:** Clean the code ([`d737d02`](https://github.com/es-ude/elastic-ai.creator/commit/d737d02122207bcd24f4b7c960b71db095d34a26))
* **vhdl:** Make linear layers better timing ([`1c6a3ae`](https://github.com/es-ude/elastic-ai.creator/commit/1c6a3aeeeaee929affbb092eb485c1cf7a323355))
* **vhdl:** Merge from main ([`fefd3ba`](https://github.com/es-ude/elastic-ai.creator/commit/fefd3ba4ab1fa8ae9d09bfc6185f906175f7a6ff))

### Fix
* **vhdl:** Fix error during integrating to a MLP model ([`0e2b89c`](https://github.com/es-ude/elastic-ai.creator/commit/0e2b89c898497f35a2ad840bd3065429799bdf61))
* **vhdl:** Fix small error in the template file ([`fe94518`](https://github.com/es-ude/elastic-ai.creator/commit/fe94518ff2e5e44f7c1ff8f9bf8b4ff8f0b5cf41))
* **vhdl:** Remove the layer name in the example file ([`767b5f9`](https://github.com/es-ude/elastic-ai.creator/commit/767b5f9c62d493d35e5a294b1363c861d5438fa5))

## v0.25.0 (2022-11-22)
### Feature
* **vhdl:** Apply the expand_template function to the already existing templates ([`c958f54`](https://github.com/es-ude/elastic-ai.creator/commit/c958f545f4c2cf2414a007753b416ec73c410458))
* **vhdl:** Add expand_template function that fills string templates instead of format strings ([`eb9ee98`](https://github.com/es-ude/elastic-ai.creator/commit/eb9ee987f73ffb26e8280ec3c32b32e38896d3c1))

### Fix
* **vhdl:** Fix the error from merging braches ([`c386766`](https://github.com/es-ude/elastic-ai.creator/commit/c386766ea654852c5ad5254cefc1fab28f544c66))

## v0.24.0 (2022-11-22)
### Feature
* **vhdl:** Add layer_id parameter to build function and set it to a unique value during translation ([`cfdf949`](https://github.com/es-ude/elastic-ai.creator/commit/cfdf9492190e24230293e3b0b1b312bfc9710952))

### Fix
* **vhdl:** Remove duplicated key ([`5a4bcd6`](https://github.com/es-ude/elastic-ai.creator/commit/5a4bcd6fb6de9cff6c639866db1dd50918f3039b))

## v0.23.0 (2022-11-15)
### Feature
* **vhdl:** Remove the previous linear_1d implementation ([`0f1b9aa`](https://github.com/es-ude/elastic-ai.creator/commit/0f1b9aa2f1c12f5c0fc1fe6a3db482f40041c057))
* **vhdl:** Enable multiple linear layers in the same model, by adding layer_name ([`3a99a30`](https://github.com/es-ude/elastic-ai.creator/commit/3a99a3059dd53b913e7d619cbce28014007bf854))
* **vhdl:** Merge main to current working branch ([`35db3c5`](https://github.com/es-ude/elastic-ai.creator/commit/35db3c56608493c6b33d05e0c2250cedb0374c8e))
* **vhdl:** Check the component interface ([`53791c5`](https://github.com/es-ude/elastic-ai.creator/commit/53791c5eb9a72793b16a0a41eb79ed8932b8e32d))
* **vhdl:** Add default build function mapping and small changes ([`b1d6f2a`](https://github.com/es-ude/elastic-ai.creator/commit/b1d6f2ac1040e63781d5f4af7ee29e486d9b6d69))
* **vhdl:** Add fp_linear build function, and test passed ([`ffcbb1d`](https://github.com/es-ude/elastic-ai.creator/commit/ffcbb1d57408ad03e91bd1228bc6d3289f1d0c66))
* **vhdl:** Add fp_linear_module and test passed ([`241fd65`](https://github.com/es-ude/elastic-ai.creator/commit/241fd652495d6ce582873f1bcc297302f3d61764))
* **vhdl:** Add fp_linear_component and its template unittest is passed ([`6e97316`](https://github.com/es-ude/elastic-ai.creator/commit/6e973168ca244e4cf407c48b31406d2eed73b4b0))

### Documentation
* **vhdl:** Change documentation ([`d3fb540`](https://github.com/es-ude/elastic-ai.creator/commit/d3fb5402c7acb09cee3df535671f22d5011f2f47))

## v0.22.0 (2022-11-13)
### Feature
* **vhdl:** Raise an exception if the build folder already exists ([`d09bfa1`](https://github.com/es-ude/elastic-ai.creator/commit/d09bfa105d909b58432cf8883ee55a6b11639add))

### Documentation
* **vhdl:** Add missing parameter in docstring of the translate_model function ([`458a02c`](https://github.com/es-ude/elastic-ai.creator/commit/458a02c38402a0860500d5821b68890fcc78c01a))

## v0.21.0 (2022-11-13)
### Feature
* **vhdl:** Add default build mapping for fp_hard_sigmoid and fp_relu ([`c9c4d9f`](https://github.com/es-ude/elastic-ai.creator/commit/c9c4d9f329ed2c56d47f2b698dbe1d3b34c1c8a5))
* **vhdl:** Add fixed point relu to translator ([`80935ce`](https://github.com/es-ude/elastic-ai.creator/commit/80935ce550a2e99267a55b41ad272906faf211a5))

## v0.20.1 (2022-11-10)
### Fix
* **vhdl:** Fix incompatible signature of the forward function ([`ff6c165`](https://github.com/es-ude/elastic-ai.creator/commit/ff6c165cd0bf17477051548018b791809fff33c9))

## v0.20.0 (2022-11-08)
### Feature
* **vhdl:** Integrate fixed point hard sigmoid to the translator ([`0a07cee`](https://github.com/es-ude/elastic-ai.creator/commit/0a07ceeb3d238456dad08448b543f4a075873322))
* **examples:** Add example using quantized modules to verify the current state of the translator ([`0c55e00`](https://github.com/es-ude/elastic-ai.creator/commit/0c55e00657c0d260766155995b75f25bff642e24))

### Documentation
* **vhdl:** Add documentation for the quantized_modules package ([`9da4a0d`](https://github.com/es-ude/elastic-ai.creator/commit/9da4a0d380304a7ab8834049ad93bed547816ddb))

## v0.19.0 (2022-11-05)
### Feature
* **vhdl:** Merge translate_model and generate_code functions ([`c12562e`](https://github.com/es-ude/elastic-ai.creator/commit/c12562ee4a55c61b5ef82b5ef37568fe32e8f525))

## v0.18.0 (2022-11-04)
### Feature
* **vhdl:** Add clamp to min or max fixed point integer for overflowing values ([`ca3fc19`](https://github.com/es-ude/elastic-ai.creator/commit/ca3fc19aec062d4de34a4698c9e0a9351b41c761))
* **examples:** Add simulated fixed point inference to the example ([`4f81d8d`](https://github.com/es-ude/elastic-ai.creator/commit/4f81d8d3d44f1c677fc1a12edf94b7b614d72efb))
* **vhdl:** Implement evaluator that evaluates a model according to a given metric ([`a0b089a`](https://github.com/es-ude/elastic-ai.creator/commit/a0b089ad1f7c32acc0c4522bf830080442e8414d))
* **vhdl:** Implement evaluator for simulation of a quantized inference ([`353e82e`](https://github.com/es-ude/elastic-ai.creator/commit/353e82e798359c3b15a42a02dcdc63e071b2d34e))
* **vhdl:** Use fixed point hard sigmoid and relu in the example ([`90350b9`](https://github.com/es-ude/elastic-ai.creator/commit/90350b91b9ac917c8c1f0ab50c2744fb09671947))
* **vhdl:** Implement a version of relu for qat and quantized inference ([`ddd9607`](https://github.com/es-ude/elastic-ai.creator/commit/ddd9607e8dbf333817112dfe24f795ac717f609e))
* **vhdl:** Fix wrong calculation of fixed point values and add quantized forward functions ([`93046d3`](https://github.com/es-ude/elastic-ai.creator/commit/93046d3b93d1a977c4106cf56e7f98847a47aa00))
* **vhdl:** Refactoring and start implementing hard sigmoid activation function ([`ff94c9d`](https://github.com/es-ude/elastic-ai.creator/commit/ff94c9dd1d1297f02e82a0d1f7f203f80c8d2732))

## v0.17.0 (2022-10-22)
### Feature
* **examples:** Visualize model parameters ([`5e1b4fc`](https://github.com/es-ude/elastic-ai.creator/commit/5e1b4fc4c827c55d19cb9bc4206f706bcc737fba))

## v0.16.0 (2022-10-22)
### Feature
* **vhdl:** Implement qat for linear layer ([`d3ba49e`](https://github.com/es-ude/elastic-ai.creator/commit/d3ba49e266b2931c1b16677dd91f17a75f091501))
* **example:** Add the ability to plot the model parameters ([`b1b0b5e`](https://github.com/es-ude/elastic-ai.creator/commit/b1b0b5e7697992c4c53825c739e2fb2dcc903dac))
* **vhdl:** Move the input, weight and output quantization to the linear layer ([`0c8b259`](https://github.com/es-ude/elastic-ai.creator/commit/0c8b259ef688c606ebd4f8486ef7b6f48e0f8713))
* **examples:** Commit current state of the fixed point linear example ([`9b8ecae`](https://github.com/es-ude/elastic-ai.creator/commit/9b8ecae971bc1dedabf17e79272008a3cbfb5123))
* **vhdl:** Add feature to automatically derive fixed point parameters from a factory ([`70618d5`](https://github.com/es-ude/elastic-ai.creator/commit/70618d512718efd7e718491af52e1acbc6c86622))
* **examples:** Start implementing an example for learning a simple logic function ([`6cff6de`](https://github.com/es-ude/elastic-ai.creator/commit/6cff6deccd5c2080e930d93f5e145e4d7ea6a41e))
* **vhdl:** Move tracing example to example folder ([`b942155`](https://github.com/es-ude/elastic-ai.creator/commit/b942155a240a6f34f0f02361b6631b431a448443))
* **vhdl:** Make base linear package private ([`b3cfa55`](https://github.com/es-ude/elastic-ai.creator/commit/b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554))
* **vhdl:** Add a type for fixed point factories ([`53b8499`](https://github.com/es-ude/elastic-ai.creator/commit/53b84991671832c2e7fa24e61d927b7c039832d9))
* **vhdl:** Implement custom linear layers that allows to do fixed point calculations ([`c2364f6`](https://github.com/es-ude/elastic-ai.creator/commit/c2364f6182bb8406e90a78d632bc868537705fd2))
* **vhdl:** Add function to get attribute names of an object matching a regex ([`acc8e29`](https://github.com/es-ude/elastic-ai.creator/commit/acc8e29e2771d5642e1371af6fb3c44f83b5ebc7))
* **vhdl:** Working further on the fixed point configuration finder ([`beb9da0`](https://github.com/es-ude/elastic-ai.creator/commit/beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d))
* **vhdl:** Start implementing fixed point evaluator ([`0f9c62a`](https://github.com/es-ude/elastic-ai.creator/commit/0f9c62a38f9df6ee4e84f1a3b5524df03511b438))
* **vhdl:** Implement signed fixed point integer to FixedPoint object ([`0a2fc79`](https://github.com/es-ude/elastic-ai.creator/commit/0a2fc7952dc13ea48c749856bf809a5540166598))
* **vhdl:** Implement automatic derivation of fixed point parameters in the lstm example ([`504008d`](https://github.com/es-ude/elastic-ai.creator/commit/504008d7ef3f402f8476bb77f02a4a37176d229e))

### Fix
* **vhdl:** Fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase ([`da11356`](https://github.com/es-ude/elastic-ai.creator/commit/da113561d69158ccc2a9266adb1eddcc79b1cb7d))

## v0.15.0 (2022-09-29)
### Feature
* **vhdl:** Implement clipped fixed point representation ([`8e53506`](https://github.com/es-ude/elastic-ai.creator/commit/8e53506fce0ba5adaa124ccd61de3b340bf1c95f))

## v0.14.0 (2022-09-28)
### Feature
* **vhdl:** Implement from_unsigned_int and from_signed_int function and remove unused function ([`aca77f5`](https://github.com/es-ude/elastic-ai.creator/commit/aca77f5eac396f21821b07706ff250b2589dd037))

### Fix
* **vhdl:** Reimplement unsigned_int_values_to_fixed_point function ([`cdd069e`](https://github.com/es-ude/elastic-ai.creator/commit/cdd069e6adffa882bb34fea2b7179891c282045b))

## v0.13.0 (2022-09-10)
### Feature
* **gh-workflow:** Explicitly set poetry version ([`82d202f`](https://github.com/es-ude/elastic-ai.creator/commit/82d202f0229e7931fc7371f69abe0d1fe3a58134))
* **vhdl:** Remove translatable protocol ([`ca52a92`](https://github.com/es-ude/elastic-ai.creator/commit/ca52a92d1bdc017773b872eaa5011b5117394472))
* **vhdl:** Remove translatable protocol ([`37412e8`](https://github.com/es-ude/elastic-ai.creator/commit/37412e87d89d16c9159cf12ef00032343119100c))

## v0.12.1 (2022-08-29)
### Fix
* **qat:** Reimplement binarize ([`9bbccdd`](https://github.com/es-ude/elastic-ai.creator/commit/9bbccddfc6ce6c2b928166cdfaf1112b294dba17))

## v0.12.0 (2022-08-26)
### Feature
* **vhdl:** Make work library name customizable ([`95fd8aa`](https://github.com/es-ude/elastic-ai.creator/commit/95fd8aa0d7e512aeb04893de2df2e58cc4b3e641))
* **vhdl:** Insert values in the updated lstm.vhd template ([`4d9dccb`](https://github.com/es-ude/elastic-ai.creator/commit/4d9dccbdb11afebb466f476c22539828bf5458b1))
* **examples:** Add linear layer to the translation example ([`5f1e1db`](https://github.com/es-ude/elastic-ai.creator/commit/5f1e1db8da7ce533cb592d56ca97e25ca563a60e))
* **vhdl:** Add an easy way to get a fixed point factory ([`d98ff03`](https://github.com/es-ude/elastic-ai.creator/commit/d98ff0351f739859ed668a2ec295421e29fd24ec))
* **vhdl:** Implement the translation of a linear1d layer ([`b627e78`](https://github.com/es-ude/elastic-ai.creator/commit/b627e780d054adcdd89009d87aa33fa31c913504))

### Fix
* **vhdl:** Remove some comments ([`13cc1a1`](https://github.com/es-ude/elastic-ai.creator/commit/13cc1a1ade14ccc7aa686523270dec20936ed14d))
* **vhdl:** Remove unused work library ([`c68fd9d`](https://github.com/es-ude/elastic-ai.creator/commit/c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0))
* **vhdl:** Fix test ([`c125bf1`](https://github.com/es-ude/elastic-ai.creator/commit/c125bf16297ee9e39660ee904ab54268e8901d48))
* **vhdl:** Add changes from Chao after testing the translator ([`5a5d532`](https://github.com/es-ude/elastic-ai.creator/commit/5a5d5325a3f598e0163d4eac0601b5961c2f5780))
* **vhdl:** Pre-add input-hidden and hidden-hidden bias ([`750941c`](https://github.com/es-ude/elastic-ai.creator/commit/750941c3150cabefa2f393f6b12105a358a70f7f))
* **vhdl:** Fix calculation of the addr_width of the linear1d layer ([`6fa2b2a`](https://github.com/es-ude/elastic-ai.creator/commit/6fa2b2a3bc83d3a51eb955d1464501662f6676a8))

### Documentation
* **vhdl:** Adapt diagrams to the latest changes ([`c1750eb`](https://github.com/es-ude/elastic-ai.creator/commit/c1750eb19f92a705f8f36ccefc9729d3545f0743))
* **readme:** Move translator documentation to the vhdl package ([`9a90949`](https://github.com/es-ude/elastic-ai.creator/commit/9a90949528978ff4732f585986a71cedd44e82a5))
* **readme:** Small changes of the documentation ([`9e7699c`](https://github.com/es-ude/elastic-ai.creator/commit/9e7699ce617581f67f85cf4ef7d945d99df241be))
* **readme:** Update documentation according the newly added linear1d layer ([`41e2486`](https://github.com/es-ude/elastic-ai.creator/commit/41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b))
* **readme:** Add documentation on how the translator works ([`91ebea3`](https://github.com/es-ude/elastic-ai.creator/commit/91ebea3fb7e7883f56b2cd9152769d151449a49a))

## v0.11.1 (2022-08-18)
### Fix
* **qat:** Remove deprecated threshold and codomain properties ([`5db9669`](https://github.com/es-ude/elastic-ai.creator/commit/5db9669fc3942851e65607a869bb822430df7836))

## v0.11.0 (2022-08-11)
### Feature
* **vhdl:** Adapt the example to the changes of the translation ([`6a5644e`](https://github.com/es-ude/elastic-ai.creator/commit/6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e))
* **vhdl:** Change translation from LSTMCell to LSTM ([`5e4f1cf`](https://github.com/es-ude/elastic-ai.creator/commit/5e4f1cff380fabd3685660a0c279b9098c4ef278))
* **vhdl:** Removed the possibility to get a build function from a type ([`dbc2e8f`](https://github.com/es-ude/elastic-ai.creator/commit/dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2))
* **vhdl:** Make build function mapping more general so that it can be reused for other frameworks ([`3369d7f`](https://github.com/es-ude/elastic-ai.creator/commit/3369d7fb6a7d08930514a7c0553c9efe65fc54b9))
* **vhdl:** Change build function mapping to a different approach ([`b1b79b2`](https://github.com/es-ude/elastic-ai.creator/commit/b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e))
* **vhdl:** Add LSTMCellTranslationArguments to __init__.py file ([`061ead4`](https://github.com/es-ude/elastic-ai.creator/commit/061ead404dc82ddc79ac75c155328ad5733eb04a))
* **vhdl:** Pass an DTO to a translatable instead of raw arguments to fix typing errors ([`2c33869`](https://github.com/es-ude/elastic-ai.creator/commit/2c33869cce5bed725a90ea3a4980bc026aec1ac4))
* **vhdl:** Pass an DTO to a translatable instead of raw arguments to fix typing errors ([`4738725`](https://github.com/es-ude/elastic-ai.creator/commit/4738725d09ca9114064c4c42dd2818fc6d5c973b))
* **vhdl:** Implement a more functional build function mapping ([`1425e03`](https://github.com/es-ude/elastic-ai.creator/commit/1425e0304cf35617106199936d3b014c0d8ca483))
* **examples:** Add an example using the vhdl translator for pytorch ([`395adcd`](https://github.com/es-ude/elastic-ai.creator/commit/395adcd3e843b7f55f6156ba183dc8800055ef51))
* **vhdl:** Add the ability to infer the build function from a given layer object or type ([`306df14`](https://github.com/es-ude/elastic-ai.creator/commit/306df1427177d15c1b1e2c59b2e774a2a6e2c471))
* **vhdl:** First untested draft for the pytorch translator ([`7e59462`](https://github.com/es-ude/elastic-ai.creator/commit/7e5946259381af397e1ccd25006815af8256026f))
* **vhdl:** Implementation of the mapping of a torch module to the corresponding build function ([`b076fa3`](https://github.com/es-ude/elastic-ai.creator/commit/b076fa32cef3c64f8fcc45df24814f4333c90b5c))
* **vhdl:** Use __init__ files to simplify the usage ([`3cc07ee`](https://github.com/es-ude/elastic-ai.creator/commit/3cc07ee048a349ef5a6a5383dcd829d64b48de2d))
* **vhdl:** Add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell ([`baca5bb`](https://github.com/es-ude/elastic-ai.creator/commit/baca5bb6c22692cf9bfc02a9147711b8869930fd))
* **vhdl:** Abstract LSTM cell takes float weights instead of FixedPoint weights ([`a5818cc`](https://github.com/es-ude/elastic-ai.creator/commit/a5818cc0edd918ef3ca49e843738823e988bfd79))
* **vhdl:** Introduce translation arguments ([`2c3a8c7`](https://github.com/es-ude/elastic-ai.creator/commit/2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f))
* **vhdl:** Add a protocol specify a translatable layer ([`0fa966e`](https://github.com/es-ude/elastic-ai.creator/commit/0fa966e7f99ef2adb19321b3ca92202616b4c0a2))
* **vhdl:** Add ability to pass kwargs to the translate function of a translatable layer ([`196812e`](https://github.com/es-ude/elastic-ai.creator/commit/196812eecd0dc49a1b8c2d6675b9018ca07e003e))
* **vhdl:** Implementation of a LSTMCell class that can be translated to VHDL ([`ace37fe`](https://github.com/es-ude/elastic-ai.creator/commit/ace37fe4b215327bc5b43344ffcd0c44a4822dda))

### Fix
* **vhdl:** Remove print call ([`55164b7`](https://github.com/es-ude/elastic-ai.creator/commit/55164b78c61f37f4cdadde0385965ee540e4f555))
* **vhdl:** Rename LSTMCell translatable to LSTM ([`e05cd04`](https://github.com/es-ude/elastic-ai.creator/commit/e05cd042daf0420b2046607e00eeef3606a6defb))
* **examples:** Use LSTMTranslationArguments object instead of a dictionary ([`98a4d97`](https://github.com/es-ude/elastic-ai.creator/commit/98a4d97f8fbd217f67ed4009ab63ccc4705f720d))
* **vhdl:** Fix mypy typing errors ([`e1dba31`](https://github.com/es-ude/elastic-ai.creator/commit/e1dba317585c269ad58719184fb4764cc66485ae))
* **vhdl:** Fix wrong pytorch lstm cell class path ([`85a733c`](https://github.com/es-ude/elastic-ai.creator/commit/85a733cb5ff821bb602b5021f6438b7d5909382e))
* **vhdl:** Fix test ([`528910c`](https://github.com/es-ude/elastic-ai.creator/commit/528910cf3fe28958ebb7b246104e83df77bbf3f4))

### Documentation
* **vhdl:** Add some docstrings to the functions of the translator ([`6f9215e`](https://github.com/es-ude/elastic-ai.creator/commit/6f9215e5fc35287517d884a702bf887d7a09aa7f))
* **readme:** Fix commands of install dev dependencies ([`870e2de`](https://github.com/es-ude/elastic-ai.creator/commit/870e2de30f48223d8005bcf1240b624ebb314ad7))

## v0.10.1 (2022-06-29)
### Fix
* **gh-workflow:** Fix error in the main.yml ([`4a6ff5e`](https://github.com/es-ude/elastic-ai.creator/commit/4a6ff5e61f35661a3ef83ce4335c109333834d6d))
* **gh-workflow:** Try to fix the error with the semantic release tool ([`bc115f8`](https://github.com/es-ude/elastic-ai.creator/commit/bc115f899bd85e720448bfa67fe9964bb56c594b))

## v0.10.0 (2022-06-26)
### Feature
* **vhdl:** Format_vhdl function blocks the process until formatting is complete ([`a8a1bd0`](https://github.com/es-ude/elastic-ai.creator/commit/a8a1bd0e7a4db075d0cef4a9eb125a860a697719))

### Documentation
* **readme:** Remove compile instructions for onnx as they are no longer needed ([`3bee70a`](https://github.com/es-ude/elastic-ai.creator/commit/3bee70abe4185a0a6708ffc998bdc74004f90b8a))
* **vhdl:** Add a docstring with an example to the FixedPoint class ([`961d766`](https://github.com/es-ude/elastic-ai.creator/commit/961d76678d366730f57bbd69b43c38124c003bf7))

## v0.9.0 (2022-06-22)
### Feature
* **vhdl:** Integrate FixedPoint type ([`b67a609`](https://github.com/es-ude/elastic-ai.creator/commit/b67a6096023a51ff4882a8cdd03a7765884c8d93))
* **vhdl:** Add function to convert FixedPoint to signed int representation ([`03001ed`](https://github.com/es-ude/elastic-ai.creator/commit/03001ed608ac934e8bbdcdfa1acb2fc7c163a89a))
* **vhdl:** Verify total bits and frac bits for multiple lists ([`360d318`](https://github.com/es-ude/elastic-ai.creator/commit/360d318db0076d9077ceb94f3f7904d95e2b12f6))
* **vhdl:** Integrate FixedPoint datatype in the LSTM test bench classes ([`7cbb88a`](https://github.com/es-ude/elastic-ai.creator/commit/7cbb88a7f77728776e0e976dcc68505b4162f0cc))
* **vhdl:** Add function to convert a list of ints to a list of FixedPoint objects ([`abece1f`](https://github.com/es-ude/elastic-ai.creator/commit/abece1fd38af607c5f5734aeacd77a1743ff3411))
* **vhdl:** Add function to convert list of float values to a list of FixedPoint objects ([`02b26d8`](https://github.com/es-ude/elastic-ai.creator/commit/02b26d868cad2a5a5bed2350a2929cf362ccdca8))
* **vhdl:** Change Rom that it uses the FixedPoint datatype ([`876cdb8`](https://github.com/es-ude/elastic-ai.creator/commit/876cdb821ff0ac67ae2345c8a36e4a742cce0949))
* **vhdl:** Add a function to infer total and frac bits from a sequence of FixedPoint values ([`9cc2b72`](https://github.com/es-ude/elastic-ai.creator/commit/9cc2b721b147628b2abf524129eeaac8f68520d5))
* **vhdl:** Separate hex/bin representation from elasticai.creator.vhdl hex/bin representation ([`eb8fe60`](https://github.com/es-ude/elastic-ai.creator/commit/eb8fe60300ee7572500f9f9d11b62a9c5abff802))

### Fix
* **vhdl:** Correct usage of the lookup_table_generator_function according to the type hints ([`9812ee8`](https://github.com/es-ude/elastic-ai.creator/commit/9812ee85cd467e261af942b30493ac0e970ea5e4))
* **vhdl:** Remove old brevitas code ([`86a8104`](https://github.com/es-ude/elastic-ai.creator/commit/86a8104cb6049dc016a5c8da08a7d2abc011935b))
* **vhdl:** Change value so that it fits into the value range of a fixed point value ([`b4e973e`](https://github.com/es-ude/elastic-ai.creator/commit/b4e973ebb8a087351e07966821229f69dc345d79))

## v0.8.0 (2022-06-08)
### Feature
* **gh-workflow:** Increase python version ([`02403e6`](https://github.com/es-ude/elastic-ai.creator/commit/02403e6cb7d8c9acc4357d9649fd2ae0834030a0))
* **pyproject:** Drop brevitas support ([`103f188`](https://github.com/es-ude/elastic-ai.creator/commit/103f1882c8da81cdf114f10b1b76c2ce89a07cba))
* Bump python version to 3.10 ([`47f5f07`](https://github.com/es-ude/elastic-ai.creator/commit/47f5f0718460a966faaa937b2c6b016720434082))

### Fix
* **pyproject:** Update poetry lock file ([`0116934`](https://github.com/es-ude/elastic-ai.creator/commit/0116934b994c4e743b1be009172de0e07acd9182))
* **pyproject:** Update poetry lock file ([`9230672`](https://github.com/es-ude/elastic-ai.creator/commit/92306722dabe5c4196e79a7cbbebab1e75ac3e6d))
* **pyproject:** Correct deps ([`7935ba1`](https://github.com/es-ude/elastic-ai.creator/commit/7935ba19bcbda7e47ddbc358c12af3aa2a01df0a))
* **precomputation:** Change ModuleProto to Module ([`cfe418e`](https://github.com/es-ude/elastic-ai.creator/commit/cfe418e41889708a53c255a8a7abcd6f1648f8f2))
* **gh-workflow:** Set more explicit python version ([`9c44093`](https://github.com/es-ude/elastic-ai.creator/commit/9c44093c6cd41d05a2d178e6e113bd10f7b86016))
* **vhdl:** Fix import of Sequence type ([`2c463ac`](https://github.com/es-ude/elastic-ai.creator/commit/2c463acdbdae0ed7dc9fa99730f53db94deb7142))
* **pyproject:** Set correct version numbers and add protobuf dependency ([`260e5fb`](https://github.com/es-ude/elastic-ai.creator/commit/260e5fb31c425ad9ba2ec31f2fa292961fd28ffa))
* Specify exact python version in github workflow ([`f3ffb18`](https://github.com/es-ude/elastic-ai.creator/commit/f3ffb183e86b722cec5efb31e0937c4810542aef))
* Fix dependencies + onnx integration tests ([`f06d0f8`](https://github.com/es-ude/elastic-ai.creator/commit/f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0))
* Resolve dependency version conflicts ([`32bd544`](https://github.com/es-ude/elastic-ai.creator/commit/32bd544b2e74b8b57497f3fd604deb5ed86ebb42))

## v0.7.0 (2022-06-05)
### Feature
* **vhdl:** Add rich comparison methods, multiple operators and a bit iterator to FixedPoint ([`116b006`](https://github.com/es-ude/elastic-ai.creator/commit/116b00647c05ef6854d3cbd1ab0f79c58f0c450d))
* **vhdl:** Start implementing FixedPoint datatype ([`8c4f420`](https://github.com/es-ude/elastic-ai.creator/commit/8c4f42097ff416f8e9056af430bda01a5bd42df5))

## v0.6.1 (2022-05-27)
### Fix
* **vhdl:** Saving generated examples to a directory instead of a giving an explicit file path ([`eb41d8d`](https://github.com/es-ude/elastic-ai.creator/commit/eb41d8db9af5171ac2826f41e98b5d85598b582d))

## v0.6.0 (2022-05-25)
### Fix
* **vhdl:** Move missing files ([`e4ae3c2`](https://github.com/es-ude/elastic-ai.creator/commit/e4ae3c2815a33b8f4f33c9578ab5cae0842277aa))
* **vhdl:** Fix previously broken imports ([`bf694f8`](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9))

### Breaking
* move modules out of generator package  ([`bf694f8`](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9))

## v0.5.0 (2022-05-25)
### Feature
* **precomputation, typing:** Make IOTable iterable and fix types ([`faa1d77`](https://github.com/es-ude/elastic-ai.creator/commit/faa1d7799bd6e8223cc4953170286d425255bb7b))
* **vhdl:** Add multiline template expansion ([`309ea35`](https://github.com/es-ude/elastic-ai.creator/commit/309ea350fae2b4e54bf06101aadc28e227d30cbb))
* **vhdl:** Add multiline template expansion ([`0d7f91f`](https://github.com/es-ude/elastic-ai.creator/commit/0d7f91f7347a5501eba02ad40499a5c0fdcce3bc))
* **vhdl:** Add multiline template expansion ([`5779708`](https://github.com/es-ude/elastic-ai.creator/commit/5779708c2de34d9c32a18af689b4b094d513ef8d))
* **vhdl:** Add multiline template expansion ([`3177fcd`](https://github.com/es-ude/elastic-ai.creator/commit/3177fcd4e5f1830608e9f6590b5af312bb74b7a9))

### Documentation
* **readme:** Add git commit message scopes ([`fe8e328`](https://github.com/es-ude/elastic-ai.creator/commit/fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f))
* **readme:** Fix table of contents and section headers ([`ecdef5d`](https://github.com/es-ude/elastic-ai.creator/commit/ecdef5da63c2c10e61a159c144c5c3707a5699e8))
* **readme:** Shorten ([`e535ea0`](https://github.com/es-ude/elastic-ai.creator/commit/e535ea0fd9d783f29ebb32d756077289d8baa8c9))

## v0.4.2 (2022-05-24)
### Fix
* **number-repr:** Fix a bug and add some parameter checks ([`a78e9e8`](https://github.com/es-ude/elastic-ai.creator/commit/a78e9e8f669c477d0629695f5c7c8ad8628f0522))

## v0.4.1 (2022-05-24)
### Fix
* Minor errors ([`812809e`](https://github.com/es-ude/elastic-ai.creator/commit/812809e1d0e706df3a0514b3503dc283ea12d7a4))

## v0.4.0 (2022-05-23)
### Feature
* **vhdl:** Allow ToLogicEncoder to register symbols in batches ([`9209279`](https://github.com/es-ude/elastic-ai.creator/commit/9209279debe651b653d2fee44533ccbdae945b32))

### Fix
* **vhdl:** Improve names in scope of ToLogicEncoder ([`67f7312`](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6))

### Breaking
* rename numerics attr to _symbols, mapping attribute to _mapping. rename add_numeric to register_symbol  ([`67f7312`](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6))

## v0.3.10 (2022-05-23)
### Fix
* **types:** Add missing mlframework types ([`3b5cf5f`](https://github.com/es-ude/elastic-ai.creator/commit/3b5cf5f8be829e109db363c25ecff76634f9d94f))
* **typing:** Fix some mypy errors ([`35b8fdf`](https://github.com/es-ude/elastic-ai.creator/commit/35b8fdf4cb0736770d9592f86499192e1e84d673))

## v0.3.9 (2022-05-22)
### Fix
* **pre-commit:** Add missing commitlint.config.js ([`2251de8`](https://github.com/es-ude/elastic-ai.creator/commit/2251de83f60823d21346aedcc2b2e9aac4c27458))
* **pyproject:** Fix duplicate field ([`6616cab`](https://github.com/es-ude/elastic-ai.creator/commit/6616cab3b0342f0b5d0b8bbdbbdf719de56d5631))
* **gh-workflow:** Updat changelog correctly ([`e76a41c`](https://github.com/es-ude/elastic-ai.creator/commit/e76a41cf55463cbc2a4ffa5b2b233d49695302b9))
* Add missing tags_utils again ([`910c611`](https://github.com/es-ude/elastic-ai.creator/commit/910c6116600b82e2c52c7d46896d92b63954d7c7))
* **gh-workflow:** Install latest isort fixing broken imports ([`a61ef44`](https://github.com/es-ude/elastic-ai.creator/commit/a61ef445672f913ec4ebc4cc8b46c2ef9099bec7))
* **gh-workflow:** Set git user+mail to gh-actions ([`174ed47`](https://github.com/es-ude/elastic-ai.creator/commit/174ed478b04b846912d6b0315f1143f24bc94524))
* **input_domains:** Add missing import of itertools ([`a6b0344`](https://github.com/es-ude/elastic-ai.creator/commit/a6b0344ac4b933112b19b8603358a0adc7274533))
* **precomputation:** Correct numpy namespace for type alias ([`a6c5842`](https://github.com/es-ude/elastic-ai.creator/commit/a6c5842920c00ae6e53e226650e0fbfe48aac44a))
* **gh-workflow:** Fix job dependencies ([`6a7d3ee`](https://github.com/es-ude/elastic-ai.creator/commit/6a7d3eeb975ca303aa30fce21bd29d14cf9982d3))
* **gh-workflow:** Fix typo ([`7f86205`](https://github.com/es-ude/elastic-ai.creator/commit/7f8620502ee544917db42ea12c7cb2eadbaef8cc))
* **gh-workflow:** Typo unit-test -> unit-tests ([`1dbd71f`](https://github.com/es-ude/elastic-ai.creator/commit/1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73))
* **gh-workflow:** Close brace ([`a6e4b99`](https://github.com/es-ude/elastic-ai.creator/commit/a6e4b999dadd163881fa96d03977c9c392a9267b))
* **gh-workflow:** Bump version ([`2cb3a72`](https://github.com/es-ude/elastic-ai.creator/commit/2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6))
* **gh-workflow:** Fix syntax error ([`895326d`](https://github.com/es-ude/elastic-ai.creator/commit/895326d67eb7ba1bb866a45c8b149778c93dc043))

### Documentation
* **readme:** Add brief explanation of pre-commit ([`3626bb0`](https://github.com/es-ude/elastic-ai.creator/commit/3626bb07cc1c8600b193bc380ae8275116ebaba8))
* Update changelog ([`e1aa8c9`](https://github.com/es-ude/elastic-ai.creator/commit/e1aa8c93554fc15c25a586b8e89eecda6dc03514))
