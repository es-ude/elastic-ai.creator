# Changelog

All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -

## [0.60.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.2..v0.60.0) - 2024-12-18

### Bug Fixes

- **(MiddlewareSpec)** transmit high byte first instead of low - ([f2bd5af](https://github.com/es-ude/elastic-ai.creator/commit/f2bd5af1cf6d9e4f40da8c89a89c61663cf12086)) - Silas Brandenburg
- **(MiddlewareSpec)** correct counter in example code - ([1860657](https://github.com/es-ude/elastic-ai.creator/commit/1860657968f0828502eecfb892c6e58fab93bf10)) - Silas Brandenburg
- **(contribution,docs)** fix tables and language - ([63a1b9d](https://github.com/es-ude/elastic-ai.creator/commit/63a1b9d42aa7a8f3866b978516a7269cec10e61b)) - Lukas Einhaus
- **(dependencies)** fixed the dependency for the runtime utils - ([cfaf318](https://github.com/es-ude/elastic-ai.creator/commit/cfaf318915a046f0b5707a56c2fcbdb9e312f1dc)) - Leo Buron
- **(dependencies)** fixed the poetry lock file - ([31868ca](https://github.com/es-ude/elastic-ai.creator/commit/31868caefb3959d0966c93d01a45c234a9041b55)) - Leo Buron
- **(firmwareEnv5)** save testbench to separate folder - ([9937431](https://github.com/es-ude/elastic-ai.creator/commit/99374317966a0db05c147bf99d322da5b14b0f5a)) - Lukas Einhaus
- **(imports)** remove toplevel __init__.py - ([c7f0a78](https://github.com/es-ude/elastic-ai.creator/commit/c7f0a7820c094789d8ae7e4bc9076c5cda167f8d)) - Lukas Einhaus
- **(ir)** fix bug where iterator was not remembering visited nodes - ([3da0dbc](https://github.com/es-ude/elastic-ai.creator/commit/3da0dbc5e84748fdd7db5ee78b9cd40636f19e7e)) - Lukas Einhaus
- **(ir)** fix conceptual problems with abstract ir data type - ([1e6210d](https://github.com/es-ude/elastic-ai.creator/commit/1e6210db742f3e1b9b2613126cc48262e6eddee4)) - Lukas Einhaus
- **(ir)** remove dead code and fix typing - ([609eb51](https://github.com/es-ude/elastic-ai.creator/commit/609eb51c45c298e190a1e6f2133623b456e9ee2c)) - Lukas Einhaus
- **(ir)** make graph iterators deterministic - ([2c3b27a](https://github.com/es-ude/elastic-ai.creator/commit/2c3b27a0e8afbf7bdbea3ce8e45abbbc65408184)) - Lukas Einhaus
- **(lstm_skeleton)** xil to work lib - ([005ed36](https://github.com/es-ude/elastic-ai.creator/commit/005ed36a4ff8bac6bb1ba1ed29e5e9cfe0be6c73)) - Lukas Einhaus
- **(nn)** fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs - ([054b8ab](https://github.com/es-ude/elastic-ai.creator/commit/054b8ab4d3569b3ae105b791ea0e8f116a8ddfd6)) - Leo Buron
- **(nn)** fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs - ([6969839](https://github.com/es-ude/elastic-ai.creator/commit/696983974ac1fee6ef25dc32072b052fb98fdc1f)) - Leo Buron
- **(nn)** linear layer uses signals now. So simulation works - ([7f5d30c](https://github.com/es-ude/elastic-ai.creator/commit/7f5d30c6c066506b66112c9ba15fe367ce33f9a8)) - Leo Buron
- **(nn)** revert changes in linear.tpl.vhd - ([f11b606](https://github.com/es-ude/elastic-ai.creator/commit/f11b6061ff88c4b16e70131508b9f10758c9b90d)) - Leo Buron
- **(nn)** fixed error in convolution - ([390656c](https://github.com/es-ude/elastic-ai.creator/commit/390656cc00cd4f827c27badae232ac8073f480a2)) - Leo Buron
- **(nn)** fixed code generation test for linear layer - ([7f8f445](https://github.com/es-ude/elastic-ai.creator/commit/7f8f4455ea1352d0c30fea05e22fb7ce561d654c)) - leo
- **(precomp)** fix incorrectly sorted inputs, add input/output widths and assert them - ([74f5e26](https://github.com/es-ude/elastic-ai.creator/commit/74f5e265fb388249a3d46c1743bc3b3e38366a78)) - Julian Hoever
- **(project)** removed init in elasticAi - ([be65e7c](https://github.com/es-ude/elastic-ai.creator/commit/be65e7c223e339b5ec03fc3b54ec3e4782a58d98)) - Leo Buron
- **(skeleton)** fix wrong signal name in integration test - ([74ebc32](https://github.com/es-ude/elastic-ai.creator/commit/74ebc32e938936d2d60c3da09773439c5675106d)) - Julian Hoever
- **(skeleton)** fix skeleton for mlp use case - ([e4b67cc](https://github.com/es-ude/elastic-ai.creator/commit/e4b67ccbc4178629f35f3f3a89259d2bfae3aba0)) - Lukas Einhaus
- **(template)** implement function only supported in python3.11 and higher - ([5223adf](https://github.com/es-ude/elastic-ai.creator/commit/5223adfdf6551f9758ee4dbdef9df1f2aed36377)) - Julian Hoever
- **(test)** add expected newline to end of skeleton - ([dcb20b7](https://github.com/es-ude/elastic-ai.creator/commit/dcb20b712945439b1c3db799404beb33d8587e4f)) - Lukas Einhaus
- **(tests)** fix wrong path that leads to temporary files created in the project tree - ([0ff9c0b](https://github.com/es-ude/elastic-ai.creator/commit/0ff9c0b04442e6da0d76f3991254cae63bf260e8)) - Julian Hoever
- **(vhdl)** fixed the test for the old skeleton and added another one for skeleton v2 - ([87b11a4](https://github.com/es-ude/elastic-ai.creator/commit/87b11a4455ec059ed8e3fdb3a405a976435facd6)) - Leo Buron
- **(vhdl)** added an exception raise for the skeleton for not supported configurations - ([11d006c](https://github.com/es-ude/elastic-ai.creator/commit/11d006c146b9eaa809460729132222a0595e6793)) - Leo Buron
- **(vhdl)** added an exception raise for the skeleton for not supported configurations - ([fc129fe](https://github.com/es-ude/elastic-ai.creator/commit/fc129fe8d83b7cacb54c615f698f42d633c73e5c)) - Leo Buron
- **(vhdl)** warn when using skeleton v1 - ([5a46331](https://github.com/es-ude/elastic-ai.creator/commit/5a4633193cf2c6db699fa19c47ddfbc53599c1fe)) - Lukas Einhaus
- **(vhdl)** #374 remove unnecessary data_buf from top module. It is not used anywhere so should have no effect - ([e8cd5a7](https://github.com/es-ude/elastic-ai.creator/commit/e8cd5a797c92b4f29b94dad1a9b7de4d090a98ae)) - Leo Buron
- **(vhdl)** fixed error in template - ([948af39](https://github.com/es-ude/elastic-ai.creator/commit/948af39669a9070518a96bd3611cb6d43b405986)) - Leo Buron
- **(vhdl)** fixed an error in the skeleton v2 template - ([dc2ee96](https://github.com/es-ude/elastic-ai.creator/commit/dc2ee964c8898a359fafe75b2bcb216ab39ebf2a)) - Leo Buron
- **(vhdl)** fixed the test for the firmware with skelton v2 - ([75ef96a](https://github.com/es-ude/elastic-ai.creator/commit/75ef96a21236c9fc3a6e830aedc5978a9e033c9e)) - Leo Buron
- **(vhdl)** fixed error in test - ([2d7f140](https://github.com/es-ude/elastic-ai.creator/commit/2d7f140bc9382be47074c1cbda3015f10ecdfaab)) - Leo Buron
- **(vhdl)** fixed test for changes in sensitivity list and for rising/falling edge clock - ([088bc1f](https://github.com/es-ude/elastic-ai.creator/commit/088bc1f7a206739701d6bf9735b3974add0262c0)) - leo
- added skeleton_1.vhd needs to be changed - ([632bf89](https://github.com/es-ude/elastic-ai.creator/commit/632bf8974ace775c8289351d03c026e587c237ed)) - Leo Buron
- update deps to resolve security issues - ([6568d28](https://github.com/es-ude/elastic-ai.creator/commit/6568d2830120922f77d2e183aa5764369143135f)) - Lukas Einhaus
- make type hints 3.10 compatible - ([db8a0f8](https://github.com/es-ude/elastic-ai.creator/commit/db8a0f8dc836e09bc4cc978e574b4d22be798954)) - Lukas Einhaus
- remove outdated srcs in skeleton plugin - ([57ae044](https://github.com/es-ude/elastic-ai.creator/commit/57ae0442099fe36a2e8c31fe100d2eba59779093)) - Lukas Einhaus

### Documentation

- **(middleware)** add register documentation - ([59f7ed4](https://github.com/es-ude/elastic-ai.creator/commit/59f7ed4c044b6062d37d66c7dc97cb31a056939b)) - David P. Federl
- **(nn)** removed unnecessary comments - ([25cdf90](https://github.com/es-ude/elastic-ai.creator/commit/25cdf904571da8d4f60418ce52447c8959b4c87b)) - leo
- **(nn)** removed unnecessary comments - ([6b1256f](https://github.com/es-ude/elastic-ai.creator/commit/6b1256f0af5bb7e857d14560d800d6455b95003a)) - leo
- **(nn)** added comments to parsing functions in testbenches - ([55c9f4d](https://github.com/es-ude/elastic-ai.creator/commit/55c9f4de6ce269f60f25edd91592aea1debe8701)) - leo
- **(nn)** added more context for the parse reported content functions - ([70c8b4b](https://github.com/es-ude/elastic-ai.creator/commit/70c8b4bbd01aaf272ccf6a91af4d91a333dce41f)) - leo
- **(skeleton)** add timing diagram to skeleton/middleware spec - ([574116b](https://github.com/es-ude/elastic-ai.creator/commit/574116b529b20334c6646de1fd20f3e95dc47218)) - Lukas Einhaus
- **(skeleton)** explain we need to read each result byte two times - ([96572fb](https://github.com/es-ude/elastic-ai.creator/commit/96572fb21958c7b79505e1ea004cdf9681e8097d)) - Lukas Einhaus
- add more middleware/skeleton specification - ([b62d982](https://github.com/es-ude/elastic-ai.creator/commit/b62d982f13360adffdfbd6a041ae30aaf83f7571)) - Lukas Einhaus
- fix hw function id length - ([3539c9f](https://github.com/es-ude/elastic-ai.creator/commit/3539c9f65a6bccf72c4cfb0312a4b2408e0b4fb9)) - Lukas Einhaus

### Features

- **(examples)** added an basic example for a network using skeleton v2 - ([6d94158](https://github.com/es-ude/elastic-ai.creator/commit/6d941584c40d41049bd27e9da8c2dc204f79080b)) - Leo Buron
- **(firmware)** new firmware that does not save testbenches - ([8ad3272](https://github.com/es-ude/elastic-ai.creator/commit/8ad3272c80a350348df1ce7a562df6f51928a4ee)) - Julian Hoever
- **(firmware)** test that firmware generates skeleton correctly - ([3a18656](https://github.com/es-ude/elastic-ai.creator/commit/3a1865642e60fcd4f4fbf49917d0930663bc38aa)) - Julian Hoever
- **(firmware)** create separate LSTMFirmwareENv5 - ([17a274c](https://github.com/es-ude/elastic-ai.creator/commit/17a274c6bb23fb5721b3e07cd16916bcbd3889c8)) - Julian Hoever
- **(ir)** add graph delegate and iterators - ([868a188](https://github.com/es-ude/elastic-ai.creator/commit/868a188aa4ce0be92608997bbfdb2916e7f8603e)) - Lukas Einhaus
- **(ir)** add graph delegate and iterators - ([1cd2a35](https://github.com/es-ude/elastic-ai.creator/commit/1cd2a353288bf08f2982bfafbfa13923da0729bb)) - Lukas Einhaus
- **(ir)** add abstract ir data class and nodes - ([bb81f0d](https://github.com/es-ude/elastic-ai.creator/commit/bb81f0dea0ee8ea8646e57d85cc070baddf91e8a)) - Lukas Einhaus
- **(ir)** introduce read only field for IrData - ([be1e8fb](https://github.com/es-ude/elastic-ai.creator/commit/be1e8fb3a659d30882b77399016fa8c21d8f0e6b)) - Lukas Einhaus
- **(ir)** make suc/predecessors sorted/deterministic - ([8825607](https://github.com/es-ude/elastic-ai.creator/commit/88256079288bae95668ffe60aced222e920419c3)) - Lukas Einhaus
- **(ir)** make suc/predecessors sorted/deterministic - ([2fd8e5e](https://github.com/es-ude/elastic-ai.creator/commit/2fd8e5e5a60fa7d7d3c69f1d8fcb0e783eade399)) - Lukas Einhaus
- **(ir)** add basic graph data structure - ([e1dfef2](https://github.com/es-ude/elastic-ai.creator/commit/e1dfef26688ffc819fe07ae7831df6b899c4b7f3)) - Lukas Einhaus
- **(ir)** add basic function registry - ([3785db0](https://github.com/es-ude/elastic-ai.creator/commit/3785db0d42f2f7a9118b9b5c3e60d1f83f9bbd86)) - Lukas Einhaus
- **(ir)** add LoweringPass class - ([d598021](https://github.com/es-ude/elastic-ai.creator/commit/d5980214c2441b52eef063bed865de2eecd52f10)) - Lukas Einhaus
- **(lstm)** set more specific return type for create_testbench function - ([7e3f54b](https://github.com/es-ude/elastic-ai.creator/commit/7e3f54b5cbc7aa6d026d60b28f8c32c05768aa0c)) - Julian Hoever
- **(nn)** added conv1d. Simulation works. End-to-end system test is still pending - ([93e5ecd](https://github.com/es-ude/elastic-ai.creator/commit/93e5ecdaae2987d34772a504bc843019419bd845)) - Leo Buron
- **(nn)** added simulation for linear layer - ([aac395b](https://github.com/es-ude/elastic-ai.creator/commit/aac395b9413943d6252bf5cb4866d96173216ae8)) - Leo Buron
- **(nn)** added conv1d. Simulation works. End-to-end system test is still pending - ([d8ca219](https://github.com/es-ude/elastic-ai.creator/commit/d8ca2193fec46161c3f77d60773ad19396a7090c)) - Leo Buron
- **(nn)** added simulation for linear layer - ([a2cd0a0](https://github.com/es-ude/elastic-ai.creator/commit/a2cd0a07854f746353009af3c63b22e03f9fcabb)) - Leo Buron
- **(nn)** added enV5 usb library to development pyproject.toml. This will be used in the future to do system tests - ([3089341](https://github.com/es-ude/elastic-ai.creator/commit/3089341849008fbfb1ff66029ea522702cc4303f)) - Leo Buron
- **(nn-qgrad)** added fixed point config, autograd and quantize - ([97bb203](https://github.com/es-ude/elastic-ai.creator/commit/97bb203898e6d689fff54c73da722584aca6882f)) - Leo Buron
- **(nn-qgrad)** added basic layers - ([82db217](https://github.com/es-ude/elastic-ai.creator/commit/82db217242fed30a955dcf7a69eb98a56e4b931a)) - Leo Buron
- **(plugin)** load plugin and call generated fn - ([0492e0b](https://github.com/es-ude/elastic-ai.creator/commit/0492e0b94eae88eb536bd7b85859527026ec273d)) - Lukas Einhaus
- **(plugin)** move plugin_loader and type_handler decorators - ([6bba61d](https://github.com/es-ude/elastic-ai.creator/commit/6bba61d5f7758f1b9db14a0938e29f4c163c52b9)) - Lukas Einhaus
- **(plugins)** load plugin description from package - ([05a99c3](https://github.com/es-ude/elastic-ai.creator/commit/05a99c3e71a3c408a4ab273b7fe453b215d39ef9)) - Lukas Einhaus
- **(plugins)** load plugin description from package - ([7dfae73](https://github.com/es-ude/elastic-ai.creator/commit/7dfae73b226d6cec7d5e0660b4da0fd78bef4439)) - Lukas Einhaus
- **(pyproject)** remove restriction to pytorch versions < 2.0.1 - ([bb47705](https://github.com/es-ude/elastic-ai.creator/commit/bb477058440e07e2bdd6c467e328219519510771)) - Julian Hoever
- **(pyproject)** allow python3.10 - ([0628024](https://github.com/es-ude/elastic-ai.creator/commit/0628024ba826ebbdbe5b5deda4aac67d81876248)) - Julian Hoever
- **(skeleton)** add general skeleton class - ([b4ffacb](https://github.com/es-ude/elastic-ai.creator/commit/b4ffacb1847685851def6beb9f53044fe5dbd75f)) - Julian Hoever
- **(skeleton_id)** tweak api for skel id computation - ([f7d9a77](https://github.com/es-ude/elastic-ai.creator/commit/f7d9a7786e06a500dafcc6cbf3f08f81083c6166)) - Lukas Einhaus
- **(skeleton_id)** move hw accel meta to dedicated module - ([9f65b8d](https://github.com/es-ude/elastic-ai.creator/commit/9f65b8dce91fcdfd87f7f1229a06c2d3776f8ad5)) - Lukas Einhaus
- **(template)** allow '${key}' placeholders for multiline templates - ([d25eef1](https://github.com/es-ude/elastic-ai.creator/commit/d25eef1369c754911e56ed5aa4a92f62b2716325)) - Julian Hoever
- **(tests)** echo server works now - ([a4359a0](https://github.com/es-ude/elastic-ai.creator/commit/a4359a0f08fa5a620ce414c8de6e133613427a65)) - Leo Buron
- **(tests)** linear layer system test with elastic node works now - ([238964a](https://github.com/es-ude/elastic-ai.creator/commit/238964a119b31db57c44085c336c8605e10c8e9a)) - Leo Buron
- **(vhdl)** added skeleton version 2 to project - ([6ed2c94](https://github.com/es-ude/elastic-ai.creator/commit/6ed2c94abbcb0d090eac3844fb59e27983f7ed11)) - Leo Buron
- **(vhdl)** added a generator for echo server with skeleton #378 - ([2c3faf5](https://github.com/es-ude/elastic-ai.creator/commit/2c3faf575df21f1aca236138257097ddd2320bff)) - Leo Buron
- **(vhdl)** added an example for the echoserver with skeleton v2 - ([3f46780](https://github.com/es-ude/elastic-ai.creator/commit/3f46780d44dc6f0d220b3a3d82f71e33ae38fdac)) - Leo Buron
- **(vhdl)** add automatic deterministic skeleton id generation - ([eb7e59f](https://github.com/es-ude/elastic-ai.creator/commit/eb7e59f7aa7506206e6807ef6a649e8b458930b4)) - Lukas Einhaus
- added a bash script to automatically build the vivado file with the help of vivado 2021.1 on a server - ([eb8c835](https://github.com/es-ude/elastic-ai.creator/commit/eb8c835529d736037b085f5ede0490ca342bac3e)) - Leo Buron
- add skeleton for sequential layer - ([34e8202](https://github.com/es-ude/elastic-ai.creator/commit/34e8202281e0be4d79d78df66cbcffc9b4db3878)) - Lukas Einhaus
- add support for less than 8 bit in skeleton - ([231f0ca](https://github.com/es-ude/elastic-ai.creator/commit/231f0ca808248b740421e5bb516b71e5f0c434ce)) - Leo Buron
- convert negative numbers to bit patterns using two's complement - ([c94dc3b](https://github.com/es-ude/elastic-ai.creator/commit/c94dc3ba59698e87eac2efe702a43dfc925401bd)) - Leo Buron
- allow python3.12 - ([46d6cfb](https://github.com/es-ude/elastic-ai.creator/commit/46d6cfb52eb2c6de471d9dd0310a9152200ec0db)) - Lukas Einhaus
- abstract class DesignCreator inherits from torch.nn.Module - ([d6e70ed](https://github.com/es-ude/elastic-ai.creator/commit/d6e70ed1c84d025bb2eebbfda45c67ad9ba1f987)) - Julian Hoever
- add plugin loader and improve function registry - ([0a8ac61](https://github.com/es-ude/elastic-ai.creator/commit/0a8ac61fef8792ab177f7635d86d4f9ae23029b1)) - Lukas Einhaus
- add basic but flexible templating component - ([2ae0506](https://github.com/es-ude/elastic-ai.creator/commit/2ae050611b9bc2cc93624e99bad7c1244dd2b6c4)) - Lukas Einhaus
- remove ir2vhdl (shouldnt have been committed) - ([20fb891](https://github.com/es-ude/elastic-ai.creator/commit/20fb8916f78ea3a78ea7eeef9af1d3f071168ca2)) - Lukas Einhaus

### Miscellaneous Chores

- **(commitlint)** add test/build/style as types - ([7d00767](https://github.com/es-ude/elastic-ai.creator/commit/7d0076794361fc99aeaaf7b80474679e8cd6d257)) - Lukas Einhaus
- **(dependency)** fixed dependency of elasticai-runtime-env5 from develop branch to specific commit - ([a5ac0df](https://github.com/es-ude/elastic-ai.creator/commit/a5ac0dfb7e0db20f21daaa75d4dd1e162f298cea)) - leo
- **(fn-registry)** remove redundant tests - ([3f0c243](https://github.com/es-ude/elastic-ai.creator/commit/3f0c243ad1f35172e199e94d3fe060b86b943661)) - Lukas Einhaus
- **(gitignore)** add devenv/direnv/uvlock - ([476cefa](https://github.com/es-ude/elastic-ai.creator/commit/476cefa326eb3084a74717c872981dbbf97feff0)) - Lukas Einhaus
- **(pipeline)** added package.json to fix the verison of commitlint - ([f8a7a0f](https://github.com/es-ude/elastic-ai.creator/commit/f8a7a0f3f888e4937baa8d0c7423637facaf443d)) - Leo Buron
- **(poetry)** synchronize pyproject.toml and poetry.lock - ([76a8baa](https://github.com/es-ude/elastic-ai.creator/commit/76a8baa01b52daf59248c33994e747f7b3dc4cb8)) - Julian Hoever
- **(pre-commit)** run black with python3.12 to support new syntax - ([ff51308](https://github.com/es-ude/elastic-ai.creator/commit/ff5130872f82206e68075f9b6c99b53dfa746a39)) - Lukas Einhaus
- **(pyproject.toml)** clean up external deps - ([2f83d46](https://github.com/es-ude/elastic-ai.creator/commit/2f83d46e0a39a079f9f3884ee71eb37a87d972c0)) - Lukas Einhaus
- only throw a warning if commit message exceeds char limit - ([3e1d509](https://github.com/es-ude/elastic-ai.creator/commit/3e1d509e23d8aa5302e708bb309514294b9d7984)) - Julian Hoever
- use python3.10 to run the tests - ([3151ba2](https://github.com/es-ude/elastic-ai.creator/commit/3151ba269b6265c1f90bf12b125f0c62e5e969f0)) - Julian Hoever
- update pre-commit (necessary to fix broken black deps) - ([9102843](https://github.com/es-ude/elastic-ai.creator/commit/9102843a898829a0dea0001f009a41265b4cf919)) - Lukas Einhaus
- create coverage report for develop as well - ([8c11c01](https://github.com/es-ude/elastic-ai.creator/commit/8c11c01ae66485d70538f07effbb637533de085f)) - Lukas Einhaus
- allow 'bump' for commit msg type - ([8b341ca](https://github.com/es-ude/elastic-ai.creator/commit/8b341caf19271235caaab3f161b1b561b6a8fbf5)) - Lukas Einhaus

### Refactoring

- **(examples)** added __init__.py - ([ad897c2](https://github.com/es-ude/elastic-ai.creator/commit/ad897c20df13c550eb8110299c2c85f3ba960eeb)) - Leo Buron
- **(ir)** decouple fn registering and calling - ([ab737b9](https://github.com/es-ude/elastic-ai.creator/commit/ab737b9bc4f86781e45d5d0b2d804ab5892a495d)) - Lukas Einhaus
- **(ir)** use new descriptor for registering fns in lowerable - ([2bd382d](https://github.com/es-ude/elastic-ai.creator/commit/2bd382ddd78212aaf925592b9c7f7838c85e89cb)) - Lukas Einhaus
- **(lowering)** avoid keeping two registries - ([21a951e](https://github.com/es-ude/elastic-ai.creator/commit/21a951ed97229f2451d4ddf28c52db794e6f86be)) - Lukas Einhaus
- **(lstm)** rename _integration_test to example - ([6828739](https://github.com/es-ude/elastic-ai.creator/commit/6828739ea5ee27cbe077b5c31ffbf14d66d5f480)) - Julian Hoever
- **(nn)** add new replacement variable in log2 calculation of linear layer - ([082b8fd](https://github.com/es-ude/elastic-ai.creator/commit/082b8fd23ba37f7428b49ab7dcbfabb056c37544)) - AErbsloeh
- **(nn)** moved mac operators to vhdl shared design - ([4925d67](https://github.com/es-ude/elastic-ai.creator/commit/4925d673e8086fceb5af31d1e577c56a003f1dd2)) - leo
- **(nn)** moved simulated layer. MAC operator design simulations do not work - ([3b927b6](https://github.com/es-ude/elastic-ai.creator/commit/3b927b693e989a4e82f97b925df28641b7b33fab)) - leo
- **(tests)** making the test a bit more convinient - ([6c778bc](https://github.com/es-ude/elastic-ai.creator/commit/6c778bca80ed20c85f2ed2d00701e3b0f2152486)) - Leo Buron
- **(tests)** moved the opening of the serial port to context manager - ([8431bc7](https://github.com/es-ude/elastic-ai.creator/commit/8431bc74cd1f8444df5a0b6f5b184e52979ffc95)) - leo
- **(vhdl)** better ghdl simulation class - ([873fd42](https://github.com/es-ude/elastic-ai.creator/commit/873fd421db4f0bfb179479c43ee459b71dbeee01)) - Leo Buron
- **(vhdl)** made the name a property so the name is already set correctly and still accessible - ([593682b](https://github.com/es-ude/elastic-ai.creator/commit/593682b933642a809496dd2a9b00fdce0e9ba19d)) - Leo Buron
- **(vhdl)** better ghdl simulation class - ([75c1cbe](https://github.com/es-ude/elastic-ai.creator/commit/75c1cbeaebf373561317af48db4c5081265e2452)) - Leo Buron
- **(vhdl)** made the name a property so the name is already set correctly and still accessible - ([b422ccb](https://github.com/es-ude/elastic-ai.creator/commit/b422ccbdc6458fcab2e99b934d53886e08f1a5be)) - Leo Buron
- **(vhdl)** changing the wake_up signal to best practice method - ([04221ec](https://github.com/es-ude/elastic-ai.creator/commit/04221ec93724939dcc3bc4d3e28428ca1afffe28)) - Leo Buron
- **(vhdl)** changed sensitivity list to clock only - ([01dd3c5](https://github.com/es-ude/elastic-ai.creator/commit/01dd3c5095d2d515cd1516b3a7ca56fe370bee6d)) - leo
- **(vhdl)** removed unnecessary print statements and added type hint - ([c3615bd](https://github.com/es-ude/elastic-ai.creator/commit/c3615bddd1a0fab13001dc2457b5f9094c7a91e7)) - leo
- remove unnecessary files - ([26b8afa](https://github.com/es-ude/elastic-ai.creator/commit/26b8afaa457969777b03f04e537470f1f6917055)) - Lukas Einhaus
- move design_creator module to nn package - ([f17a6da](https://github.com/es-ude/elastic-ai.creator/commit/f17a6dac5e612ba99bfc906862de3e3048aa7a17)) - Julian Hoever
- rename the module design_creator to design_creator_module - ([49d3ab4](https://github.com/es-ude/elastic-ai.creator/commit/49d3ab4e1d244c13d4994df65dda2f47a846aaad)) - Julian Hoever

### Style

- beautify 903501083d6acaee8b472f22e7bf24cddb3647b8 - ([cc6fda5](https://github.com/es-ude/elastic-ai.creator/commit/cc6fda534289fdf4359c46e9e08ba984c7638a07)) - github-actions
- beautify b61c6180a56de314e569338eebbbdbe45a889f42 - ([ce92f2b](https://github.com/es-ude/elastic-ai.creator/commit/ce92f2b593d3b705e34867a8b87d90e3f4a7d9a9)) - github-actions

### Build

- add missing pytest-cov - ([1707f07](https://github.com/es-ude/elastic-ai.creator/commit/1707f07eb91693a03e3e33e1f3a4cc1ab12037c2)) - Lukas Einhaus
- clean up pyproject - ([fa27806](https://github.com/es-ude/elastic-ai.creator/commit/fa27806b53b5d27f84c4ccee57e7abefbe242cf6)) - Lukas Einhaus
- omit *_test in main folder for coverage - ([da8104d](https://github.com/es-ude/elastic-ai.creator/commit/da8104da6da54438104cd6bdecd68cc06d08cadd)) - Lukas Einhaus

### Bump

- 0.59.2 -> 0.60.0 - ([a38a0ce](https://github.com/es-ude/elastic-ai.creator/commit/a38a0ce001a397ce73d8e6f86fb839258ceaddc4)) - Lukas Einhaus

### Ci

- publish on pushing release tags - ([4b16c98](https://github.com/es-ude/elastic-ai.creator/commit/4b16c98ed20599385e49a49e44a995cf4ea73dc6)) - Lukas Einhaus
- install only testing group for unit-tests job - ([73b3bf8](https://github.com/es-ude/elastic-ai.creator/commit/73b3bf822e298fde23a045d4297ff3ca48773383)) - Lukas Einhaus
- perform coverage when running checks - ([75b520f](https://github.com/es-ude/elastic-ai.creator/commit/75b520f300e4ac0bd36ed08c7c92ace70d44c64c)) - Lukas Einhaus
- don't install lsp deps in test pipeline - ([07853aa](https://github.com/es-ude/elastic-ai.creator/commit/07853aa973d5e290d869b96c3168ed5bda1cde7d)) - Lukas Einhaus
- remove publishing to test.pypi.org - ([44d50d4](https://github.com/es-ude/elastic-ai.creator/commit/44d50d446b8bc6a1f41a563fc3cb52ad61bf04fe)) - Lukas Einhaus
- remove old release workflow - ([8d0b5af](https://github.com/es-ude/elastic-ai.creator/commit/8d0b5afc48bbee40e00924860075a6fa6641fea6)) - Lukas Einhaus
- fix triggering release on tag push - ([cc5c5ef](https://github.com/es-ude/elastic-ai.creator/commit/cc5c5efaa3c185abb2e6f75e8339121afbc4f5fc)) - Lukas Einhaus
- remove errornous check for tag from release pipeline - ([1847b55](https://github.com/es-ude/elastic-ai.creator/commit/1847b550cfa700891ce1fc77f15b1358588b3238)) - Lukas Einhaus
- fix repo name in release workflow - ([ffc8529](https://github.com/es-ude/elastic-ai.creator/commit/ffc85292dceeafe56e752fc522d6e732fd7d57c2)) - Lukas Einhaus

### Wip

- **(nn)** added simulation for linear layer. Still not finished - ([67382df](https://github.com/es-ude/elastic-ai.creator/commit/67382df946ee5d12907dbf7080ea24586ddbfdb7)) - Leo Buron
- **(nn)** added simulation for linear layer. Still not finished - ([70f9b5e](https://github.com/es-ude/elastic-ai.creator/commit/70f9b5e868f154d2ee16ddf25f1ef90bb1f94f92)) - Leo Buron
- **(nn)** echoserver does not work. Linear also not - ([a673890](https://github.com/es-ude/elastic-ai.creator/commit/a673890a141943ef1006297a2cfd91bf98fb9d2e)) - Leo Buron
- **(nn)** fixed error in linear layer - ([7943b3b](https://github.com/es-ude/elastic-ai.creator/commit/7943b3b1f46cdef9b3f3d6c2d287ec3d8a1713bf)) - Leo Buron
- **(tests)** added system test for linear layer. Still work in progress - ([d94e2ad](https://github.com/es-ude/elastic-ai.creator/commit/d94e2add91ab0c669a8a3db2ae487803d2e56d34)) - Leo Buron
- **(tests)** still trying to fix linear layer - ([cd4c180](https://github.com/es-ude/elastic-ai.creator/commit/cd4c18022d6420b11735af1eb42b7270a16948b7)) - Leo Buron
- **(tests)** added conv1d and fixed stuff - ([9316e3d](https://github.com/es-ude/elastic-ai.creator/commit/9316e3d7d3a41ecdf89941abe4ed88433020fdf0)) - Leo Buron
- **(tests)** changed workflow for generating data and checking data from device - ([44ef433](https://github.com/es-ude/elastic-ai.creator/commit/44ef43307bbb268991c07001bf229faa19d6f212)) - AErbsloeh
- **(vhdl)** mac operator simulations do not work - ([3e6502c](https://github.com/es-ude/elastic-ai.creator/commit/3e6502c56d103c9f6b47133a4b8f5f9a14b0d5bc)) - leo
- contribution docs - ([35f2de2](https://github.com/es-ude/elastic-ai.creator/commit/35f2de275d7241ee6cfb9448b9896a873ee06f66)) - Lukas Einhaus
- add skeleton+middleware spec - ([adba799](https://github.com/es-ude/elastic-ai.creator/commit/adba79902ded0a3d2ccd02b9cd503be30508abf8)) - Lukas Einhaus
- update skeleton specification - ([7a95f7c](https://github.com/es-ude/elastic-ai.creator/commit/7a95f7c5ed66fadf20adefc3c59df236d8857d33)) - Lukas Einhaus
- start timing diagram for middleware/skeleton spec - ([0f51559](https://github.com/es-ude/elastic-ai.creator/commit/0f515598e52beec666ff3a34207e727b6b258536)) - Lukas Einhaus
- prepare convolution hw impl - ([374d966](https://github.com/es-ude/elastic-ai.creator/commit/374d966f2f2b1367e0e7c629b12638e8c3db0bb7)) - Lukas Einhaus

---
## [0.59.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.1..v0.59.2) - 2023-10-06

### Bug Fixes

- copy model to cpu for quantized inference - ([0c5d88e](https://github.com/es-ude/elastic-ai.creator/commit/0c5d88e26e55eb11d2a729c5a7bf6b865927b61f)) - Julian Hoever

---
## [0.59.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.59.0..v0.59.1) - 2023-10-06

### Bug Fixes

- **(lstm)** skeleton naming - ([21d057d](https://github.com/es-ude/elastic-ai.creator/commit/21d057d2497d817c09c12e90f43047aeed71e6d8)) - Lukas Einhaus
- **(precomputed)** set step_lut as non trainable parameter - ([95954c2](https://github.com/es-ude/elastic-ai.creator/commit/95954c2cbd52a85f762128ce3a88259085431536)) - Julian Hoever
- **(precomputed)** do not copy input to cpu - ([586b774](https://github.com/es-ude/elastic-ai.creator/commit/586b77458e9529dc8f12023fbefb6a3747fd222e)) - Julian Hoever
- **(tests)** add step_lut to state_dict - ([e18f46f](https://github.com/es-ude/elastic-ai.creator/commit/e18f46f0a70b1000e8e6d0ea3ecdddce2ad325d5)) - Julian Hoever
- turn `vhdl.top` into a package - ([2a272ea](https://github.com/es-ude/elastic-ai.creator/commit/2a272ea4e0b470bf8684098cf48e8121ee92d27f)) - Lukas Einhaus
- add saving constraints and sources to plug&play ENV5 - ([41aa4f1](https://github.com/es-ude/elastic-ai.creator/commit/41aa4f105fb59bcdc76b031800af916eb0c76f35)) - Lukas Einhaus
- added missing file for network skeleton tpl - ([38fe8a7](https://github.com/es-ude/elastic-ai.creator/commit/38fe8a7337479e5b7c02d66d682431e95b03e190)) - Leo Buron
- parametrize names - ([c13576d](https://github.com/es-ude/elastic-ai.creator/commit/c13576d380c58f658c4054a645f8407041a95faf)) - Lukas Einhaus
- fix lstm names - ([da279f7](https://github.com/es-ude/elastic-ai.creator/commit/da279f742409b9c31d690d385c4d04896e3afbb4)) - Lukas Einhaus
- fix lstm test bench file name - ([f879b4d](https://github.com/es-ude/elastic-ai.creator/commit/f879b4de526bc7c7cd93371b5874c1eac2f465f5)) - Lukas Einhaus
- correct `create_testbench` for lstm - ([e82af52](https://github.com/es-ude/elastic-ai.creator/commit/e82af52217f0ef1874abfcd0b43f1d905ed3e4bb)) - Lukas Einhaus
- move `create_testbench` to correct class - ([53bc568](https://github.com/es-ude/elastic-ai.creator/commit/53bc5684682abbc721969f357b0810175a89a25f)) - Lukas Einhaus
- names and templates for lstm - ([3ad358c](https://github.com/es-ude/elastic-ai.creator/commit/3ad358c698bca447376b910bdd275bc806eb6db6)) - Lukas Einhaus
- remove unnecessary instance name templ variables - ([7da4b5a](https://github.com/es-ude/elastic-ai.creator/commit/7da4b5a473e9976392eda1d4e686cc4ff9b12d0d)) - Lukas Einhaus
- don't save uut in testbench - ([7f09a2a](https://github.com/es-ude/elastic-ai.creator/commit/7f09a2ab3549d309759e531dd5b6ec4051a9d3e7)) - Lukas Einhaus
- add skeleton, etc. to generated files - ([2bbd588](https://github.com/es-ude/elastic-ai.creator/commit/2bbd588ceaec6408f61f48c2f61289c90afffef9)) - Lukas Einhaus
- fix fxp mac test - ([baad73b](https://github.com/es-ude/elastic-ai.creator/commit/baad73b8e97f5c7b65e52a1c9755eb2086de02aa)) - Lukas Einhaus
- fix skeleton test - ([12b7c27](https://github.com/es-ude/elastic-ai.creator/commit/12b7c274d1d9116d06be711066f2d5ee1cf5725e)) - Lukas Einhaus
- use linear layer name - ([9d48f09](https://github.com/es-ude/elastic-ai.creator/commit/9d48f098e60ead2854af61a6c1394e824e762538)) - Lukas Einhaus

### Documentation

- explain relationship between LSTM, LSTMNetwork and their sw/hw impl - ([7db8974](https://github.com/es-ude/elastic-ai.creator/commit/7db8974e11a586e60c7f154e7cbbbf27b75a9c41)) - Lukas Einhaus

### Features

- reintegrate lstm implementation - ([9440fbb](https://github.com/es-ude/elastic-ai.creator/commit/9440fbb1ed9c81218e62f4e60917127e128d3856)) - Lukas Einhaus
- added skeleton, middleware, top module and Vivado constraints fopr env5 LSTM example - ([67117a4](https://github.com/es-ude/elastic-ai.creator/commit/67117a443032cbafd3bcf8abcab7177a801fd659)) - Leo Buron
- add lstm_network testbench - ([37d7921](https://github.com/es-ude/elastic-ai.creator/commit/37d79212f52b949634f7af321e7b7bc56306ffeb)) - Lukas Einhaus
- inject network to FirmwareENv5 - ([bf2c53f](https://github.com/es-ude/elastic-ai.creator/commit/bf2c53f554bfc312875f08cb99d95e028364667b)) - Lukas Einhaus

### Wip

- refactor lstm - ([397499e](https://github.com/es-ude/elastic-ai.creator/commit/397499ebf4f5c4ee9c2703751a31f6373c9c24be)) - Lukas Einhaus

---
## [0.59.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.58.0..v0.59.0) - 2023-10-02

### Bug Fixes

- **(chore)** exclude simulations from coverage - ([3b31a45](https://github.com/es-ude/elastic-ai.creator/commit/3b31a45bccb262d60a1d920618457514a2bd8a95)) - Lukas Einhaus
- **(chore)** exclude simulations from coverage - ([de6a293](https://github.com/es-ude/elastic-ai.creator/commit/de6a29324c6e688139a82bea0b08afda1f0388bc)) - Lukas Einhaus
- remove need for csv in testbench - ([f905bbf](https://github.com/es-ude/elastic-ai.creator/commit/f905bbf6e1f9119677b7f73caf58b35343e0d7bb)) - Lukas Einhaus
- ignore one more line for ghdl out parsing - ([c48ec8f](https://github.com/es-ude/elastic-ai.creator/commit/c48ec8f42f906222223b7282c677ffdfe5cd06ec)) - Lukas Einhaus
- make mac impl use round to zero logic - ([f8c674f](https://github.com/es-ude/elastic-ai.creator/commit/f8c674f5a2382a1d8b9b1630a8af35977dc0c9c9)) - Lukas Einhaus

### Documentation

- add documentation for number conversion - ([617e2c6](https://github.com/es-ude/elastic-ai.creator/commit/617e2c61fab2b21459a1a62c65c184cdbcc35e09)) - Lukas Einhaus

### Features

- lstm reintegration - ([0ffa9b0](https://github.com/es-ude/elastic-ai.creator/commit/0ffa9b029636ab36570019e9f99fd5c788281e26)) - Julian Hoever
- lstm reintegration - ([c23aa53](https://github.com/es-ude/elastic-ai.creator/commit/c23aa532d02e096f3ca7a83c38a46b6fb6d295d7)) - Silas Brandenburg
- add number_conversion - ([1fde323](https://github.com/es-ude/elastic-ai.creator/commit/1fde32308a83a3de71af736eda47a81bb5bc468b)) - Lukas Einhaus
- parse ghdl output - ([3f7e05f](https://github.com/es-ude/elastic-ai.creator/commit/3f7e05f44f86776ea01b28b176116730d94a9354)) - Lukas Einhaus
- handle colons in ghdl sim parsing - ([f485531](https://github.com/es-ude/elastic-ai.creator/commit/f485531881b0874eb14df58ccb3873889cb1cac6)) - Lukas Einhaus
- basic fxp mac + hw/sw simulation tests - ([f34a1ed](https://github.com/es-ude/elastic-ai.creator/commit/f34a1edc70c34d456daa222bed537d793ee0c29e)) - Lukas Einhaus
- add xnor-popcount based mac bin impl - ([6a63eb3](https://github.com/es-ude/elastic-ai.creator/commit/6a63eb358ce3279bcdbca468aed25445ab0be13e)) - Lukas Einhaus

### Miscellaneous Chores

- add simulation test tag - ([f7fda58](https://github.com/es-ude/elastic-ai.creator/commit/f7fda587861e83c729eaa90791b34efc4f9b433d)) - Lukas Einhaus

### Refactoring

- simplify number conversion implementations - ([715d6c7](https://github.com/es-ude/elastic-ai.creator/commit/715d6c7540748fb53e93f78782f653ee45e6bdb4)) - Lukas Einhaus
- simplify simulating test benches - ([b607211](https://github.com/es-ude/elastic-ai.creator/commit/b6072118e1d84e59b1faac73ec3c75c6aca88ee9)) - Lukas Einhaus
- rename hw_integ_test.py - ([8324d1a](https://github.com/es-ude/elastic-ai.creator/commit/8324d1abaad92a12d5fcdee9925c58b9c9743aff)) - Lukas Einhaus
- move number conversion modules - ([ed3086c](https://github.com/es-ude/elastic-ai.creator/commit/ed3086c149388876e8cd243cd535199bded8f9f5)) - Lukas Einhaus
- add create_simulation method to layer - ([6c93c81](https://github.com/es-ude/elastic-ai.creator/commit/6c93c81b5910dfcc2eb71d64c137be5f9b8d0fad)) - Lukas Einhaus
- add create_simulation method to layer - ([d3f8746](https://github.com/es-ude/elastic-ai.creator/commit/d3f874636e2c78d584e7b39e31f10d0ba6ab9e9b)) - Lukas Einhaus

### Style

- beautify f94e16fd03d289124dd20dd844776d517fb91e4a - ([db35406](https://github.com/es-ude/elastic-ai.creator/commit/db354066168885e5ae91d18efd705d239319b81a)) - github-actions

### Wip

- **(hwsim)** parse ghdl output - ([3ba3d6d](https://github.com/es-ude/elastic-ai.creator/commit/3ba3d6dea2a575a4f4318323afeaa6b174b73d8c)) - Lukas Einhaus
- **(hwsim)** return computation from testbench - ([4f3c8e3](https://github.com/es-ude/elastic-ai.creator/commit/4f3c8e31c5fa327e769a129b4570e7587b1c60ac)) - Lukas Einhaus
- add min/max values for number representations - ([db6b222](https://github.com/es-ude/elastic-ai.creator/commit/db6b222da8ef41c0fec22538337d4249e1c9b779)) - Lukas Einhaus
- simulation integration test - ([7f71160](https://github.com/es-ude/elastic-ai.creator/commit/7f7116016112262f7008dc8da53e3912e867aa3a)) - Lukas Einhaus
- simulation integration test - ([c00b76a](https://github.com/es-ude/elastic-ai.creator/commit/c00b76a78b1a8ad33f9bc6bc56f32e37c07b0086)) - Lukas Einhaus
- run first simulation test with pytest - ([c374f9f](https://github.com/es-ude/elastic-ai.creator/commit/c374f9f38a8d371cd185bc99e21dfda6aa456090)) - Lukas Einhaus
- mac implementation - ([de68bae](https://github.com/es-ude/elastic-ai.creator/commit/de68baeec51834d3ac3a55dfdc5772d133bf1656)) - Lukas Einhaus

---
## [0.58.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.57.1..v0.58.0) - 2023-09-29

### Bug Fixes

- remove unnecessary output quantization of the SiLU base module - ([350faa5](https://github.com/es-ude/elastic-ai.creator/commit/350faa52d93994738a68cc0222dfb907b5174f12)) - Julian Hoever
- remove stride and padding as supported parameters for conv1d - ([05b57d1](https://github.com/es-ude/elastic-ai.creator/commit/05b57d1d2e112a21259fa2221f2204b3f6d87bfe)) - Julian Hoever
- remove already dropped padding, stride and dilation - ([79494fe](https://github.com/es-ude/elastic-ai.creator/commit/79494fe6116921ff7b4d1287bd67451ba7498ecd)) - Julian Hoever

### Documentation

- **(readme)** use create_design function instead of translate in minimal example - ([b9351ca](https://github.com/es-ude/elastic-ai.creator/commit/b9351ca28b37a0ee7f34187b01986c3ba11c6827)) - Julian Hoever

### Features

- **(linear)** use bias as default - ([14b01be](https://github.com/es-ude/elastic-ai.creator/commit/14b01be0e3aad2315daa1864588701ce2fd8dff7)) - Julian Hoever
- **(tests)** add tests for conv1d layer - ([c645297](https://github.com/es-ude/elastic-ai.creator/commit/c6452975ef02d6c2a46ca4550311238a46917636)) - Julian Hoever
- **(tests)** add tests for the conv1d design - ([c2c94bd](https://github.com/es-ude/elastic-ai.creator/commit/c2c94bd91e09e507120cfcf71e28b0797dca419c)) - Julian Hoever
- **(tests)** add tests for fixed point linear layer - ([2e959b6](https://github.com/es-ude/elastic-ai.creator/commit/2e959b619ea9ef66dfede042131f7116b18f3532)) - Julian Hoever
- **(tests)** add tests for linear design - ([55a366a](https://github.com/es-ude/elastic-ai.creator/commit/55a366a34049708ee62d0c440f67640435716900)) - Julian Hoever
- **(tests)** add small integration test to verify that linear layer creates correct design - ([f71e43f](https://github.com/es-ude/elastic-ai.creator/commit/f71e43f92cbfbf2c7a735a610c3dbca790ea8299)) - Julian Hoever
- **(tests)** add small integration test to verify that conv1d layer generates correct design - ([da45cc3](https://github.com/es-ude/elastic-ai.creator/commit/da45cc30b481ee27e9bd9d5d172e29a0bd0b519f)) - Julian Hoever
- add parameter getters - ([3c184c0](https://github.com/es-ude/elastic-ai.creator/commit/3c184c075cdb94ffae05ea7424e33dd98a4c09f9)) - Julian Hoever

### Refactoring

- **(conv1d)** remove failing test - ([e56a07f](https://github.com/es-ude/elastic-ai.creator/commit/e56a07feb90b8853f3b2901503bbe034fa7b4f16)) - Julian Hoever
- **(linear)** split layer.py into multiple files to improve readability - ([37128c9](https://github.com/es-ude/elastic-ai.creator/commit/37128c95d0a28268b1889783b31e536074d01ab9)) - Julian Hoever
- **(linear)** remove unused import - ([24164b0](https://github.com/es-ude/elastic-ai.creator/commit/24164b0ff52f9e26a21a2342c9be2d5079a270e6)) - Julian Hoever
- **(silu)** add test for the fixed point silu - ([2b0da94](https://github.com/es-ude/elastic-ai.creator/commit/2b0da947ff8cddf282294f4139a74b4b723cc4cb)) - Julian Hoever
- **(tests)** remove not necessary fixed weights - ([d9482ce](https://github.com/es-ude/elastic-ai.creator/commit/d9482cec23e20d4b370be8037fb528552b92c2cf)) - Julian Hoever
- rename SiLUWithTrainableScaleBeta to AdaptableSiLU - ([fdf34b0](https://github.com/es-ude/elastic-ai.creator/commit/fdf34b0c04eae35af3867c8f9109cf24400b4d33)) - Julian Hoever
- split batch normed conv1d and conv1d layers in seperate files and add parameter getters - ([924e5f3](https://github.com/es-ude/elastic-ai.creator/commit/924e5f3646c151667c881360ed68aad890dc5a67)) - Julian Hoever
- rename Translatable protocol to DesignCreator - ([e60bf7f](https://github.com/es-ude/elastic-ai.creator/commit/e60bf7f0688ba6e875d821c8cab4c34f4054cdec)) - Julian Hoever
- rename translatable module to design_creator - ([0aa4d72](https://github.com/es-ude/elastic-ai.creator/commit/0aa4d72458aef4a91c87334048c357a276627d3d)) - Julian Hoever

### Style

- beautify 4b05c4a847ee33048b9552a93f816d6fec3c404f - ([c10028b](https://github.com/es-ude/elastic-ai.creator/commit/c10028bd5a2019c8de67f95bf1c25c930c5ec8d0)) - github-actions

### Wip

- **(tests)** start adding tests for the BatchNormedConv1d layer - ([b53d369](https://github.com/es-ude/elastic-ai.creator/commit/b53d3694994591fdb52355ca4abe1b17af8fc2d5)) - Julian Hoever

---
## [0.57.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.57.0..v0.57.1) - 2023-08-29

### Bug Fixes

- try to exclude test files from build - ([f282ac0](https://github.com/es-ude/elastic-ai.creator/commit/f282ac06aae45451d3787d74cda54b51e7f28200)) - Julian Hoever

---
## [0.57.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.56.0..v0.57.0) - 2023-08-29

### Features

- global math operations depend on math operations of supported layers - ([127ffdb](https://github.com/es-ude/elastic-ai.creator/commit/127ffdb29853587e4f819d75077e524e7a168bc5)) - Julian Hoever
- exclude tests from build - ([72b8e0a](https://github.com/es-ude/elastic-ai.creator/commit/72b8e0af0e3fddc154be5763b26f5174cc49d7f4)) - Julian Hoever

### Refactoring

- move unit tests to the elasticai package - ([bb0ab8b](https://github.com/es-ude/elastic-ai.creator/commit/bb0ab8b8b8636c07318bae5e662836a07b5f33ec)) - Julian Hoever
- rename test files from test_*.py to *_test.py to improve readabilitly - ([b7d3557](https://github.com/es-ude/elastic-ai.creator/commit/b7d3557338617c90b9b70459c4eeff12cc1c4623)) - Julian Hoever

---
## [0.56.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.2..v0.56.0) - 2023-08-28

### Bug Fixes

- outdated imports - ([650a71e](https://github.com/es-ude/elastic-ai.creator/commit/650a71e36f9c12db1160b86f82ad4d37715b19d7)) - Lukas Einhaus
- remove deprecated layers - ([d55c041](https://github.com/es-ude/elastic-ai.creator/commit/d55c041e0141fe30bf3038074c061704ed057682)) - Lukas Einhaus
- remove deprecated layers - ([59cdaae](https://github.com/es-ude/elastic-ai.creator/commit/59cdaaea412f2d7aa6a8fd63a8ba931688ddc855)) - Lukas Einhaus
- fix broken imports and other errors that leads to failing tests - ([6eac561](https://github.com/es-ude/elastic-ai.creator/commit/6eac56189e1e449d378867a4b4d8003967f48689)) - Julian Hoever

### Documentation

- start glossary - ([b2d82cd](https://github.com/es-ude/elastic-ai.creator/commit/b2d82cdcc663d13492b611a700321c4bbcf452be)) - Lukas Einhaus
- update minimal example to reflect the most recent changes - ([8f122a0](https://github.com/es-ude/elastic-ai.creator/commit/8f122a04699c42ff7abb7179d3cb7412cf94c0ef)) - Julian Hoever
- add glossary entry - ([8855e02](https://github.com/es-ude/elastic-ai.creator/commit/8855e0248e97a2628c1ff73e69538c969f52d685)) - Julian Hoever

### Features

- add layers to __init__ file in fixed_point package to improve usability - ([93917d8](https://github.com/es-ude/elastic-ai.creator/commit/93917d8654794b7f5baa005e989f2984d6c846e3)) - Julian Hoever
- add a public quantize function to allow initial quantization of model inputs - ([c8a170c](https://github.com/es-ude/elastic-ai.creator/commit/c8a170cb1ba529855790dcaf2dad97c38171174e)) - Julian Hoever

### Miscellaneous Chores

- remove fixed commit scopes - ([9f6b2e7](https://github.com/es-ude/elastic-ai.creator/commit/9f6b2e793e7a952f5ddff7efc89677a2f00c935e)) - Julian Hoever

### Refactoring

- remove mlframework/typing.py - ([6858537](https://github.com/es-ude/elastic-ai.creator/commit/6858537f7906b75495376828c4713690b14cb461)) - Lukas Einhaus
- remove empty folders, start docs improvements - ([28a9a2d](https://github.com/es-ude/elastic-ai.creator/commit/28a9a2d96521fbcb53ca889b746470bb99aef20f)) - Lukas Einhaus
- improve separation of core packages - ([b5f469f](https://github.com/es-ude/elastic-ai.creator/commit/b5f469f1fbb354547560c9fefcd88e271382ae91)) - Lukas Einhaus
- restructure packages - ([2fa7a4f](https://github.com/es-ude/elastic-ai.creator/commit/2fa7a4f58a481868edca1bdd3a568686130873dd)) - Lukas Einhaus
- move batchnormed layers to their base versions - ([b1e0feb](https://github.com/es-ude/elastic-ai.creator/commit/b1e0feb6e45fd4f300065397ab2698382135c4b5)) - Lukas Einhaus
- move batchnormed layers to their base versions - ([cf4f1d5](https://github.com/es-ude/elastic-ai.creator/commit/cf4f1d5974e6d2201320bc6d2e017745870a14e0)) - Lukas Einhaus
- separate interface for conv1d - ([76ba0ac](https://github.com/es-ude/elastic-ai.creator/commit/76ba0ac054d7acd07ace2eb9875e9bd3473eeca3)) - Lukas Einhaus
- rename and move modules to fit our new scheme - ([8effe1a](https://github.com/es-ude/elastic-ai.creator/commit/8effe1ac03fceebd324e4ad07f9d305c8e7d0c08)) - Julian Hoever
- remove arithmetics and autograd_functions from base_modules - ([1e23c74](https://github.com/es-ude/elastic-ai.creator/commit/1e23c7406ffb0c0ab0a62aa31f8d61a502a4886f)) - Julian Hoever
- rename inputs parameter to x in forward parameter lists - ([314b747](https://github.com/es-ude/elastic-ai.creator/commit/314b747e72509a2d34e72cc1d7738e2e26c18bd3)) - Julian Hoever
- adapt the structure of the tests directory to the latest changes - ([b3cd5cc](https://github.com/es-ude/elastic-ai.creator/commit/b3cd5cc5c3cb68e7b5adb8136322426830d6db40)) - Julian Hoever
- reformat code - ([1fb5ed6](https://github.com/es-ude/elastic-ai.creator/commit/1fb5ed6663424a98a36db3ad6ef899d62c74b75c)) - Julian Hoever
- reformat code - ([3c728a0](https://github.com/es-ude/elastic-ai.creator/commit/3c728a020ffc29c052d60b8917b7721399a05766)) - Julian Hoever

### Style

- beautify fcb153ea3aa32a73e07dd1f71d148634698a6cda - ([6515ab0](https://github.com/es-ude/elastic-ai.creator/commit/6515ab0225bd4b55b4e8a7ad1a5e4acb2d397ea3)) - github-actions

---
## [0.55.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.1..v0.55.2) - 2023-08-12

### Bug Fixes

- add dummy batch dimension to meet the requirements of the batch norm - ([0f499f0](https://github.com/es-ude/elastic-ai.creator/commit/0f499f048c0606de3e14163f16e8bf049708e6f1)) - Julian Hoever

---
## [0.55.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.55.0..v0.55.1) - 2023-08-11

### Bug Fixes

- fix non existing in_channels variable and remove unused import - ([0e73c2d](https://github.com/es-ude/elastic-ai.creator/commit/0e73c2dc6876772d0caba46639af77bd5ac53b62)) - Julian Hoever

---
## [0.55.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.54.0..v0.55.0) - 2023-08-11

### Bug Fixes

- typing and errors - ([af6f859](https://github.com/es-ude/elastic-ai.creator/commit/af6f85913fd6111bcc7164a106a9cbb8d4b7b9a0)) - Silas Brandenburg

### Features

- implemented batch normed conv1d layer - ([cd6836c](https://github.com/es-ude/elastic-ai.creator/commit/cd6836cc72b3fecee1b522f9b8934fabefd46d63)) - Silas Brandenburg

### Refactoring

- set alias for Design/FPLinear - ([559395c](https://github.com/es-ude/elastic-ai.creator/commit/559395cde0dca73344cd162df04fea510a621b49)) - Silas Brandenburg

---
## [0.54.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.53.0..v0.54.0) - 2023-08-09

### Bug Fixes

- use same bit width for all rom values - ([cd609e6](https://github.com/es-ude/elastic-ai.creator/commit/cd609e65f306e62110fbdc4113f4bb330f960f19)) - Julian Hoever

### Refactoring

- rename precomputed monotonic increasing module - ([ab8dfdf](https://github.com/es-ude/elastic-ai.creator/commit/ab8dfdf4c19646ae14dd787203a380eda47c281d)) - Julian Hoever

---
## [0.53.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.52.0..v0.53.0) - 2023-08-02

### Bug Fixes

- fix missing parameter in tests for conv1d - ([d8f8d4c](https://github.com/es-ude/elastic-ai.creator/commit/d8f8d4c40ec1576c5dc58a38b2b80d9d4130b4fd)) - Silas Brandenburg

### Features

- implement fixed point one dimensional convolution - ([2ea9389](https://github.com/es-ude/elastic-ai.creator/commit/2ea9389a37eac7be62e26a9727b8824b47fc2085)) - Silas Brandenburg

### Refactoring

- simplify string - ([de8d3ec](https://github.com/es-ude/elastic-ai.creator/commit/de8d3ec98a6105d48630a0b2e6d82f15c3e75a9e)) - Silas Brandenburg

---
## [0.52.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.51.0..v0.52.0) - 2023-08-02

### Features

- implemeted base module for SiLU aka Swish activation function - ([93b5954](https://github.com/es-ude/elastic-ai.creator/commit/93b59544c1d164de2a9f9362f0aefe1aaae8d7d8)) - Silas Brandenburg
- added nn module for Swish activation function - ([b7579c9](https://github.com/es-ude/elastic-ai.creator/commit/b7579c9f1111521064a7fd0366647da7a45e2d7a)) - Silas Brandenburg
- added Swish activation function precomputed - ([fd487b5](https://github.com/es-ude/elastic-ai.creator/commit/fd487b57bb7d3e7525f935f1533f815e58f1dc0d)) - Silas Brandenburg
- added Swish activation function precomputed - ([26d292e](https://github.com/es-ude/elastic-ai.creator/commit/26d292e83183ac0b6bee7afa70f3d616e42b2438)) - Silas Brandenburg

### Refactoring

- changed names of learnable parameters in the swish function - ([bb2b7a8](https://github.com/es-ude/elastic-ai.creator/commit/bb2b7a81bf6365a54590575f39a447b6cd769cd9)) - Silas Brandenburg
- delete files that aren´t necessary - ([53aebf3](https://github.com/es-ude/elastic-ai.creator/commit/53aebf3702c9c3511ef81e8fd9a1fcca018bf26d)) - Silas Brandenburg
- removed unnecessary file - ([fe58c0f](https://github.com/es-ude/elastic-ai.creator/commit/fe58c0f22d38284e289542b6f3e58fbff60963f9)) - Silas Brandenburg
- deleted unnecessary File test_silu.py from branch - ([80e8919](https://github.com/es-ude/elastic-ai.creator/commit/80e8919078ba32dd9af0146a94bd38b63bc761b1)) - Silas Brandenburg

---
## [0.51.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.50.0..v0.51.0) - 2023-07-28

### Bug Fixes

- remove not implemented jvp function - ([0ea4834](https://github.com/es-ude/elastic-ai.creator/commit/0ea48341c02d116dd3ef2a94e0997ce8e0641b60)) - Julian Hoever
- try to fix semantic release - ([0eab187](https://github.com/es-ude/elastic-ai.creator/commit/0eab187389b3d435be473671d4a593ead8586e78)) - Julian Hoever

### Features

- rename custom float to float to improve readability - ([794bfe7](https://github.com/es-ude/elastic-ai.creator/commit/794bfe79a6a821050b33cc246e9e1cad09e7e682)) - Julian Hoever
- add debug messages - ([ee09864](https://github.com/es-ude/elastic-ai.creator/commit/ee09864686d87471617aae4ae65118096d31a6ff)) - Julian Hoever
- enable debug messages - ([36ca597](https://github.com/es-ude/elastic-ai.creator/commit/36ca597ded38bb3c5343e872ed7cf9cb09065a6f)) - Julian Hoever
- seperate semantic release run into multiple steps - ([475b425](https://github.com/es-ude/elastic-ai.creator/commit/475b425910c4a124c34ca9a68fd5c49b4789541b)) - Julian Hoever
- apply proposed semantic release migration procedure - ([d5ea981](https://github.com/es-ude/elastic-ai.creator/commit/d5ea981cd8852e5790c77d9667187168c34c81e3)) - Julian Hoever
- increase semantic release version to v8.0.4 - ([bb29612](https://github.com/es-ude/elastic-ai.creator/commit/bb2961243f20ade3e7c4a142601f58fca6e9b5ad)) - Julian Hoever
- revert changes and explicitly set semantic release version to v7 instead of v8 - ([2ecf0db](https://github.com/es-ude/elastic-ai.creator/commit/2ecf0db3c22ce034c4f36a26c96027f0229a4bf0)) - Julian Hoever

### Refactoring

- remove noise comment - ([f6be240](https://github.com/es-ude/elastic-ai.creator/commit/f6be240b03484876627f5f7de5198fd1332d6ba7)) - Julian Hoever
- remove newline - ([33fa0a9](https://github.com/es-ude/elastic-ai.creator/commit/33fa0a932b5c2126a004b429702bcda72e696069)) - Julian Hoever

---
## [0.50.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.49.0..v0.50.0) - 2023-07-11

### Bug Fixes

- return wrong number of values in the backward pass - ([6bcfa4e](https://github.com/es-ude/elastic-ai.creator/commit/6bcfa4eff8d9b7c0c0461a61800ef68ef6b0cb62)) - Julian Hoever

### Features

- implement RoundToCustomFloat autograd function - ([0794a8e](https://github.com/es-ude/elastic-ai.creator/commit/0794a8e900d6f87edc03dbd71162e7300e13b5ae)) - Julian Hoever
- implement custom float arithmetics - ([b72713e](https://github.com/es-ude/elastic-ai.creator/commit/b72713e3db1e15957e865ed95216a2f180523114)) - Julian Hoever

### Refactoring

- rename FloatArithmetics to TorchArithmetics - ([5cd7a3b](https://github.com/es-ude/elastic-ai.creator/commit/5cd7a3b6913b14456f524a9486bac2c42dc72412)) - Julian Hoever
- rename CustomFloatArithmetics to FloatArithmetics - ([824b029](https://github.com/es-ude/elastic-ai.creator/commit/824b029971789c951a243937d942d7597225e829)) - Julian Hoever

---
## [0.49.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.48.1..v0.49.0) - 2023-07-01

### Documentation

- complete table of contents - ([cf0ef63](https://github.com/es-ude/elastic-ai.creator/commit/cf0ef63eb628521f14406fb7d59cee53c71c8d60)) - Julian Hoever
- add minimal example that demonstrates the usage of the creator - ([64030f2](https://github.com/es-ude/elastic-ai.creator/commit/64030f2eb129ff8275022ab0b8bf4945d42626a8)) - Julian Hoever

### Features

- update readme and add small improvements - ([8f2bbd0](https://github.com/es-ude/elastic-ai.creator/commit/8f2bbd093e18c15421abab20ecb0f9afbc6d12a1)) - Julian Hoever

---
## [0.48.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.48.0..v0.48.1) - 2023-06-24

### Bug Fixes

- only create coverage reports in PR - ([1bd728f](https://github.com/es-ude/elastic-ai.creator/commit/1bd728f4e8edb6595a35dafd71c5d68263a7358f)) - Julian Hoever

---
## [0.48.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.2..v0.48.0) - 2023-06-24

### Bug Fixes

- use poetry run to run pytest - ([7058e42](https://github.com/es-ude/elastic-ai.creator/commit/7058e42cc7fa0849841578f2bafd6a3fc6155f2a)) - Julian Hoever

### Features

- use binary values instead of hex values to fill the rom template - ([af56c02](https://github.com/es-ude/elastic-ai.creator/commit/af56c02da42433c2db1a9a2a6ddb3705d213d765)) - Julian Hoever
- add pytest-cov dependency - ([a737729](https://github.com/es-ude/elastic-ai.creator/commit/a7377290ffee7359f6f8c0392960d7038fe2a73b)) - Julian Hoever
- add coverage workflow to create reports - ([3f6caca](https://github.com/es-ude/elastic-ai.creator/commit/3f6caca6a626923ec3d8078320fa9b70092495ee)) - Julian Hoever
- only trigger coverage report when pushing to main - ([b4b23c9](https://github.com/es-ude/elastic-ai.creator/commit/b4b23c988803165895c14a8357427a3069f09233)) - Julian Hoever

### Refactoring

- improve readability - ([e4de568](https://github.com/es-ude/elastic-ai.creator/commit/e4de5682419829675a92ff95f8e853dc28cf181e)) - Julian Hoever
- remove unused to_vhdl_hex_string function - ([24ccbf1](https://github.com/es-ude/elastic-ai.creator/commit/24ccbf1a1d9ff3d270faba19581a6f72eadb751e)) - Julian Hoever

---
## [0.47.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.1..v0.47.2) - 2023-06-23

### Bug Fixes

- fix error when passing a cuda tensor to the IdentityStepFunction - ([7f49617](https://github.com/es-ude/elastic-ai.creator/commit/7f496171a547bae17c69976c35d437428022447f)) - Julian Hoever

---
## [0.47.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.47.0..v0.47.1) - 2023-06-16

### Bug Fixes

- remove wrongly committed files - ([4fdea0c](https://github.com/es-ude/elastic-ai.creator/commit/4fdea0c9ff2db5e8af3f208bbd83d995332d5b85)) - Julian Hoever

### Miscellaneous Chores

- add do_not_commit path to prevent files from being committed by mistake - ([af13e16](https://github.com/es-ude/elastic-ai.creator/commit/af13e1687f57fc3545d0c114263ed439b78973cd)) - Julian Hoever

### Refactoring

- merge fp quant and fp dequant into a roundtofixedpoint autograd function - ([b986a62](https://github.com/es-ude/elastic-ai.creator/commit/b986a62ea7a0a58e6479aa5082ddd2de11ed27d7)) - Julian Hoever

---
## [0.47.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.46.1..v0.47.0) - 2023-06-16

### Features

- simplify project structure - ([81cbcb3](https://github.com/es-ude/elastic-ai.creator/commit/81cbcb343b26473290609c7715051059127a924b)) - Julian Hoever

### Refactoring

- remove unused manifest module - ([55f8e6d](https://github.com/es-ude/elastic-ai.creator/commit/55f8e6deac74d953b97b031a22e0dd9a73ecf20c)) - Julian Hoever

### Style

- beautify 7beebdbc67074dc6f8e8a0320563385ee49a7915 - ([1c7fead](https://github.com/es-ude/elastic-ai.creator/commit/1c7feadd0825be6648702c7ecffcdb1c2ce974f5)) - github-actions

---
## [0.46.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.46.0..v0.46.1) - 2023-06-13

### Bug Fixes

- fix wrong port definitions - ([9a4c8af](https://github.com/es-ude/elastic-ai.creator/commit/9a4c8af6f8f8be2bf6fff49c25fc0ca12cbea45a)) - Julian Hoever

---
## [0.46.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.45.0..v0.46.0) - 2023-06-13

### Bug Fixes

- fix some syntax errors - ([3997bbd](https://github.com/es-ude/elastic-ai.creator/commit/3997bbdb134a94defd4e32ad1a2eb3aa236d6b96)) - Julian Hoever
- quantize weights before inference - ([61153e6](https://github.com/es-ude/elastic-ai.creator/commit/61153e60d6854bacf0bd2501d96efc3f6e62714e)) - Julian Hoever

### Features

- add the ability to sum over dimension - ([c45c0e6](https://github.com/es-ude/elastic-ai.creator/commit/c45c0e676e1df70bf99c4c943874168781ef2a93)) - Julian Hoever
- test that conv1d uses different arithmetics - ([7eb01db](https://github.com/es-ude/elastic-ai.creator/commit/7eb01dbaa2afbbb02410e6fc6272ba02fec7878a)) - Julian Hoever
- add conv1d function to arithmetics - ([1cab190](https://github.com/es-ude/elastic-ai.creator/commit/1cab1901e324eb100f1cbccf6d54fae429210b33)) - Julian Hoever
- use conv1d arithmetics function to implement conv1d module - ([69778be](https://github.com/es-ude/elastic-ai.creator/commit/69778be7fd1becab2ad5099ebb8d64d4a0db0de5)) - Julian Hoever

### Refactoring

- remove debug print call - ([f85172e](https://github.com/es-ude/elastic-ai.creator/commit/f85172ebddfceb98a7c661cd3f57db60b19b61c0)) - Julian Hoever
- improve readablility - ([004c736](https://github.com/es-ude/elastic-ai.creator/commit/004c736cab22b4e8eed5eb867c203b4b62e7e235)) - Julian Hoever
- remove redundant tests - ([c828d53](https://github.com/es-ude/elastic-ai.creator/commit/c828d536110205b3e00f61a33e31d0cae1eaee6f)) - Julian Hoever

### Wip

- start implementing conv1d base module - ([3c535df](https://github.com/es-ude/elastic-ai.creator/commit/3c535df421b61e02a60df8735cfc14bf5c243190)) - Julian Hoever
- add padding to conv1d base module - ([54c0838](https://github.com/es-ude/elastic-ai.creator/commit/54c08387d9f0193f1651e6633d8207e6c8c03b0b)) - Julian Hoever

---
## [0.45.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.44.0..v0.45.0) - 2023-06-10

### Bug Fixes

- fix broken import in base template generator and move it with its template to own folder - ([9eb1f70](https://github.com/es-ude/elastic-ai.creator/commit/9eb1f70cff10e075712d5bf7e3fc9fcfed2aae19)) - Julian Hoever

### Features

- simplify usage for the elasticai.creator.nn.vhdl package by adding layers to __init__ - ([2c7c968](https://github.com/es-ude/elastic-ai.creator/commit/2c7c96858ec9d935389a960baee46e8c506f9b5c)) - Julian Hoever

### Miscellaneous Chores

- update dependencies - ([a2558b5](https://github.com/es-ude/elastic-ai.creator/commit/a2558b5649d5416c730cfdfebdd4d38ce48a6a88)) - Julian Hoever

### Refactoring

- remove unused template resources - ([d58f267](https://github.com/es-ude/elastic-ai.creator/commit/d58f267772839df6c254b9d749b8e5653b9a20e1)) - Julian Hoever
- rename sequential layer module according to our convention - ([ae1da5e](https://github.com/es-ude/elastic-ai.creator/commit/ae1da5e5aced255e38f0c13691a1d42f90dd5cb3)) - Julian Hoever
- remove unused and redundant port definition - ([b376b75](https://github.com/es-ude/elastic-ai.creator/commit/b376b757f6dd0e6400813688a2dfdf6ca392a6f9)) - Julian Hoever
- rename template and remove some newlines - ([707310b](https://github.com/es-ude/elastic-ai.creator/commit/707310b3202ec1b48f847a228455f8cd77436219)) - Julian Hoever
- remove some newlines, use create_port function and fix wrong template - ([1bc4a70](https://github.com/es-ude/elastic-ai.creator/commit/1bc4a70f173c9f380a76438842ba6708d1659aad)) - Julian Hoever
- transform bdd test to pytest test - ([475ec7b](https://github.com/es-ude/elastic-ai.creator/commit/475ec7bd12ed0f43b65438a2ef62aa97d3ca8b14)) - Julian Hoever
- remove unused pytest-bdd dependency - ([e9203a0](https://github.com/es-ude/elastic-ai.creator/commit/e9203a0223ef3adfcbd40af841e569438684e1c8)) - Julian Hoever
- rename monotonously increasing scalar function - ([baff8b2](https://github.com/es-ude/elastic-ai.creator/commit/baff8b2fd8569c60b51906458c2d541e1371f111)) - Julian Hoever
- better separation of designs and modules - ([44f22ae](https://github.com/es-ude/elastic-ai.creator/commit/44f22ae25a02c0c4810e64c970cdc5dd28135c89)) - Julian Hoever
- create rom design folder - ([9e40f5f](https://github.com/es-ude/elastic-ai.creator/commit/9e40f5fa9b40c2542e4ef99cf02d1b6004ad2a60)) - Julian Hoever
- remove deprecated documentation - ([349a9f8](https://github.com/es-ude/elastic-ai.creator/commit/349a9f866e001ce0494f9876d894ef0c5833817d)) - Julian Hoever
- remove unused base signal definition - ([14dc275](https://github.com/es-ude/elastic-ai.creator/commit/14dc275beea7f3c757433eb9b3872c895fc6fca3)) - Julian Hoever
- rename ports module to port_definitions - ([b5a64b8](https://github.com/es-ude/elastic-ai.creator/commit/b5a64b812145a34dd1dd0d20cb2ca31f18804a1f)) - Julian Hoever
- use container types from collections.abc instead of typing because they are deprecated - ([7a45e67](https://github.com/es-ude/elastic-ai.creator/commit/7a45e672cdcc47b426a57a8297febc8aa9664744)) - Julian Hoever
- remove unused imports - ([e8881d3](https://github.com/es-ude/elastic-ai.creator/commit/e8881d31e11d3e27489322deabb3c29d420e568b)) - Julian Hoever
- use Identity class from base_modules instead of torch - ([8f179f0](https://github.com/es-ude/elastic-ai.creator/commit/8f179f0bbbb2510b294665e0502715b6b69346c8)) - Julian Hoever

---
## [0.44.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.43.0..v0.44.0) - 2023-06-09

### Bug Fixes

- port def and impl of monotonous function design - ([2d423d4](https://github.com/es-ude/elastic-ai.creator/commit/2d423d46faa86fbf43cb8ba1d01aafe92c5bfa23)) - Lukas Einhaus
- use new Sequential constructor - ([6bb111b](https://github.com/es-ude/elastic-ai.creator/commit/6bb111b748567502c23a48a52d7e477645969996)) - Lukas Einhaus

### Refactoring

- cleanup imports - ([c402a03](https://github.com/es-ude/elastic-ai.creator/commit/c402a031f5996c6f7a1b3a5199e1cf9697e7dc5a)) - Lukas Einhaus

### Style

- beautify 95ca25571e9757d932a45749e9cf92531c13ab36 - ([cdf44ce](https://github.com/es-ude/elastic-ai.creator/commit/cdf44cec1a9a656ce6b3a9d19a717a9e7163d1b6)) - github-actions

---
## [0.43.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.42.0..v0.43.0) - 2023-06-09

### Bug Fixes

- increase default sampling intervall - ([07620d3](https://github.com/es-ude/elastic-ai.creator/commit/07620d3e2ee9db1bc6aa081a15274cb79b5ee4b0)) - Julian Hoever
- use elsif in lookup table - ([f375ba3](https://github.com/es-ude/elastic-ai.creator/commit/f375ba3784bf92887e689f77f592dfc2fa2c7e2c)) - Julian Hoever
- set correct signal names for x and y address - ([5354a2a](https://github.com/es-ude/elastic-ai.creator/commit/5354a2a0e85bc0788f5d74377c1a685e9d0e0de7)) - Julian Hoever

### Features

- introduce FPMonotonouslyIncreasingModule to easily add new activations - ([b78c922](https://github.com/es-ude/elastic-ai.creator/commit/b78c9225f7f70ec329bee5705c11d9e7b1392c41)) - Julian Hoever
- add tests for the FPMonotonouslyIncreasingModule - ([9ba64ae](https://github.com/es-ude/elastic-ai.creator/commit/9ba64ae3d253db76a6368c5e561ce28bcec2aab5)) - Julian Hoever

### Refactoring

- move all arithmetics to arithmetics folder in base_modules - ([de0fd46](https://github.com/es-ude/elastic-ai.creator/commit/de0fd460eae7d7d155188d2e73dd4cc82b913718)) - Julian Hoever
- remove unnecessary tests - ([c0756b3](https://github.com/es-ude/elastic-ai.creator/commit/c0756b3d7a7468aa0e3d7c55e126170790bae076)) - Julian Hoever

### Wip

- start implementing sigmoid activation function - ([04a99b2](https://github.com/es-ude/elastic-ai.creator/commit/04a99b2e0ff20ed89af2b8cf4d1ccdb4b7551a39)) - Julian Hoever

---
## [0.42.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.41.0..v0.42.0) - 2023-06-08

### Features

- make sure that inplace parameter is fixed defined - ([79b7a1e](https://github.com/es-ude/elastic-ai.creator/commit/79b7a1eea0cb71f5a838cfebf02970927410f594)) - Julian Hoever
- add working hardsigmoid implementation - ([db03ff0](https://github.com/es-ude/elastic-ai.creator/commit/db03ff080f878c9b9fe54303ead97c673022f3a1)) - Julian Hoever
- reimplement hard tanh activation function - ([9b86f9d](https://github.com/es-ude/elastic-ai.creator/commit/9b86f9d440cc991d624a6f3492a3caf7419bdbf3)) - Julian Hoever

---
## [0.41.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.40.0..v0.41.0) - 2023-06-08

### Features

- add fixed point ReLU module - ([62c1555](https://github.com/es-ude/elastic-ai.creator/commit/62c15557fc515c89644c674aef9fc39d22ab672f)) - Julian Hoever

---
## [0.40.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.39.0..v0.40.0) - 2023-06-04

### Bug Fixes

- fix missing creation of a subpath in the save_to function - ([2a4dbdf](https://github.com/es-ude/elastic-ai.creator/commit/2a4dbdf2f6fce4de567281002dd4640ff3ae54ed)) - Julian Hoever
- fix that last io pair was dropped when calling save_to function - ([2bc46ac](https://github.com/es-ude/elastic-ai.creator/commit/2bc46ac9c535b65ef7a3dc5cbe12b27d253c3b37)) - Julian Hoever

### Features

- add a function to easily compare tensors with pytest - ([24e737e](https://github.com/es-ude/elastic-ai.creator/commit/24e737eaea48044df3e8addaca0d1cc804a3b6f4)) - Julian Hoever
- implement autograd fn to map inputs to a subset of inputs - ([26c6ec7](https://github.com/es-ude/elastic-ai.creator/commit/26c6ec7a203eea4fed4c3eb3d5c3e4893acb545f)) - Julian Hoever
- rename autograd function and pass step lut to autograd function - ([d607e98](https://github.com/es-ude/elastic-ai.creator/commit/d607e98bd14dfa1ae23e9726b2046baaede21361)) - Julian Hoever
- pass step lut to identity step function and improve readablility - ([c1b6747](https://github.com/es-ude/elastic-ai.creator/commit/c1b67473c33ddc27590068472dcff6969f9e7135)) - Julian Hoever
- implement bufferless component interface for precomputed scalar function - ([f701a57](https://github.com/es-ude/elastic-ai.creator/commit/f701a57db54e0d5f3e5e43047725b28646cb5f15)) - Julian Hoever
- add quantized tanh implementation with lookup tables - ([3a1fb10](https://github.com/es-ude/elastic-ai.creator/commit/3a1fb10944e566ca33e3e745b939b6700421fdb9)) - Julian Hoever
- improve performance of the identity step autograd function - ([46f036c](https://github.com/es-ude/elastic-ai.creator/commit/46f036c8fb2d007d21e32214ac92d4d9aa2fe9d1)) - Julian Hoever
- simplify the use of the sequential layer (same as in torch) - ([9fad15d](https://github.com/es-ude/elastic-ai.creator/commit/9fad15d774f3573fb26f168295f9bd2ae5cdd046)) - Julian Hoever

### Refactoring

- move torch dependency to base_moduels - ([06d1aca](https://github.com/es-ude/elastic-ai.creator/commit/06d1aca6e3ca95a1e371253aa97dee831119250c)) - Julian Hoever
- remove unused base modules - ([97d1e7d](https://github.com/es-ude/elastic-ai.creator/commit/97d1e7dbc181fc03562ccbcde976eb9e661c381e)) - Julian Hoever
- small change of the folder structure - ([58783a8](https://github.com/es-ude/elastic-ai.creator/commit/58783a83a891d85c50c43a6af2ac3efa3e634657)) - Julian Hoever
- remove unnecessary tests - ([23f78db](https://github.com/es-ude/elastic-ai.creator/commit/23f78db7aec7efeef669a32ebe76ea3ebcb6b133)) - Julian Hoever
- remove default sampling intervall - ([9d7caea](https://github.com/es-ude/elastic-ai.creator/commit/9d7caeae98408d2eaf0c97032dae0b5b4b312429)) - Julian Hoever
- remove unused import - ([4de2055](https://github.com/es-ude/elastic-ai.creator/commit/4de205551938c7a284af78b5c2c418fdf95358f6)) - Julian Hoever
- change indentations - ([d5f5bf0](https://github.com/es-ude/elastic-ai.creator/commit/d5f5bf07b85d7b1902d474975da58d29bc615f6d)) - Julian Hoever
- remove the leading underscore of the class name - ([6643bf1](https://github.com/es-ude/elastic-ai.creator/commit/6643bf13dfbe50f7b98c0a49a238041c49fa8b89)) - Julian Hoever

---
## [0.39.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.38.0..v0.39.0) - 2023-05-19

### Bug Fixes

- correct tuple type annotation - ([f0e7da0](https://github.com/es-ude/elastic-ai.creator/commit/f0e7da0cf186015004970102f2b9b57a9f839585)) - Lukas Einhaus
- adjust tests to follow previous change - ([c328bd5](https://github.com/es-ude/elastic-ai.creator/commit/c328bd565d6ba84a9d1fab788051c3e884ea2094)) - Lukas Einhaus
- remove obsolete parsing functionality - ([7f85d05](https://github.com/es-ude/elastic-ai.creator/commit/7f85d05aa3da2e0fd7c266bfc9c1aad573adecc4)) - Lukas Einhaus
- children of sequential layer determine signal widths - ([3dd5c0c](https://github.com/es-ude/elastic-ai.creator/commit/3dd5c0cc4f7a52c7b3a86cec437005b86aa0a509)) - Lukas Einhaus
- remove dequantize - ([c111022](https://github.com/es-ude/elastic-ai.creator/commit/c111022854ce6965b705b3a3de296e032d7ff107)) - Julian Hoever
- allow to set affine and bias equals false in translate function - ([b351284](https://github.com/es-ude/elastic-ai.creator/commit/b351284335a77caec838a8f4ea57684e429cc35b)) - Julian Hoever

### Features

- **(template)** make precomputed scalar functions bufferless - ([89986fa](https://github.com/es-ude/elastic-ai.creator/commit/89986fad041c89d0543fe9a22946e5f5f49e2b61)) - Lukas Einhaus
- port expansion/template based on autowiring protocol - ([0d14618](https://github.com/es-ude/elastic-ai.creator/commit/0d146181c8b789b09871af43654ca2d83ea55ddb)) - Lukas Einhaus
- add basic vhdl parsing - ([5df2a3f](https://github.com/es-ude/elastic-ai.creator/commit/5df2a3ff4e9ba7ec33398a267cd983ad886d1fe7)) - Lukas Einhaus
- add standalone parser module - ([5a9b141](https://github.com/es-ude/elastic-ai.creator/commit/5a9b141285fefecf61f581417061428cda382ad5)) - Lukas Einhaus
- support parsing partial files - ([8170012](https://github.com/es-ude/elastic-ai.creator/commit/817001208b774e57cfb27fb4d4ee9d704541c9f8)) - Lukas Einhaus
- support parsing partial files - ([f2c2eb6](https://github.com/es-ude/elastic-ai.creator/commit/f2c2eb69ceb8a0b02c1c4617511ccb1528931e23)) - Lukas Einhaus
- add intermediate symbols to rule definitions - ([624b310](https://github.com/es-ude/elastic-ai.creator/commit/624b310fc9beb130902fdf3269e3f30714fe0c3f)) - Lukas Einhaus
- add AutoWirer - ([f4159c8](https://github.com/es-ude/elastic-ai.creator/commit/f4159c800fe54cc0fe73fbebdf2ac0410ddac635)) - Lukas Einhaus
- check for autowiring protocol violation - ([3f17e00](https://github.com/es-ude/elastic-ai.creator/commit/3f17e002e050dc92516e4ff5468041f06ebd6760)) - Lukas Einhaus
- add experimental precomputed tanh in fixed point - ([0e76d03](https://github.com/es-ude/elastic-ai.creator/commit/0e76d03b6d0f23d8932b94bb7728cbeea2de0289)) - Lukas Einhaus
- implement batch normed linear layer - ([9322f6f](https://github.com/es-ude/elastic-ai.creator/commit/9322f6f699f9884273c3f9815b9a026c9f7840ae)) - Julian Hoever

### Refactoring

- remove obsolete module - ([5adc999](https://github.com/es-ude/elastic-ai.creator/commit/5adc999c3f4fb5a45e569680fa466694127688da)) - Lukas Einhaus
- make identity layer/design names more specific - ([0aed47e](https://github.com/es-ude/elastic-ai.creator/commit/0aed47ebd3dbd784156a949822b8fc7c117e07c0)) - Lukas Einhaus
- remove obsolete test helper code - ([17e4e12](https://github.com/es-ude/elastic-ai.creator/commit/17e4e1250c1b94b3f72ac9dba57f7ee66825f381)) - Lukas Einhaus
- pull up tokenize functions - ([ace6f1e](https://github.com/es-ude/elastic-ai.creator/commit/ace6f1eb5d0162d7454d56a5baf6f3fb59f3dc06)) - Lukas Einhaus
- pull up parse function - ([1b8f187](https://github.com/es-ude/elastic-ai.creator/commit/1b8f1874eff63130e71c1754257d5bb3d05bb827)) - Lukas Einhaus
- move sequential layer to nn.vhdl - ([caea325](https://github.com/es-ude/elastic-ai.creator/commit/caea325588f8c87cc28d5df248129b0e73111e3d)) - Lukas Einhaus
- move binarize autograd function to autograd_functions folder - ([03d5bc8](https://github.com/es-ude/elastic-ai.creator/commit/03d5bc86462b36be30c2887593360ec48a908ab1)) - Julian Hoever
- rename FPLinear1d design to FPLinear - ([238f167](https://github.com/es-ude/elastic-ai.creator/commit/238f1671a28b9b5735ca7e01360d4dda7122a2a7)) - Julian Hoever
- remove redundant quantize function - ([02094cf](https://github.com/es-ude/elastic-ai.creator/commit/02094cf412f2846821c9c2925bedcdc585fe8a8d)) - Julian Hoever

### Wip

- add base (bufferless) identity layers - ([59b119d](https://github.com/es-ude/elastic-ai.creator/commit/59b119dc8bae5bdc7d850fad4b82d350e56e4957)) - Lukas Einhaus
- start implementing a batch normed linear layer - ([cadbd56](https://github.com/es-ude/elastic-ai.creator/commit/cadbd568d3d7acc1ac92109f6c68e46c300b1ad0)) - Julian Hoever

---
## [0.38.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.2..v0.38.0) - 2023-05-09

### Bug Fixes

- fix not inserted process name - ([dbabea0](https://github.com/es-ude/elastic-ai.creator/commit/dbabea07c888a5309d9ca55cd2c01ae0debea57d)) - Julian Hoever
- add variable - ([229d452](https://github.com/es-ude/elastic-ai.creator/commit/229d452d0c2f798ee1dd0124f50be8f01d69ede4)) - Julian Hoever
- remove broken lstm implementation - ([c524ca2](https://github.com/es-ude/elastic-ai.creator/commit/c524ca20cc49333007c4e0bbfa167912580e5c01)) - Julian Hoever

### Features

- write function of InMemoryFile and OnDiskFile now takes Template object - ([a867ea1](https://github.com/es-ude/elastic-ai.creator/commit/a867ea15980b8ca1390327f2999c4d7b91ef3041)) - Julian Hoever
- add function to get all unfilled variables of a template - ([d635cb6](https://github.com/es-ude/elastic-ai.creator/commit/d635cb6098735b451aea259a8a6f15619bfcd64f)) - Julian Hoever
- add check that all variables are filled when saving a template - ([c988d2b](https://github.com/es-ude/elastic-ai.creator/commit/c988d2bc203790ba8ab900e8a2de6996b22d6fcb)) - Julian Hoever

### Miscellaneous Chores

- remove unused workflow - ([dd08e08](https://github.com/es-ude/elastic-ai.creator/commit/dd08e08b0af74c4d7ba927c892de6081717657db)) - Julian Hoever

### Refactoring

- temporarily rename template class - ([6fb83a2](https://github.com/es-ude/elastic-ai.creator/commit/6fb83a2d773bb474bf96f4c248de8537f91673aa)) - Julian Hoever
- rename TemplateConfig protocol to Template - ([33d01ee](https://github.com/es-ude/elastic-ai.creator/commit/33d01eef31e7c9cb919a9684150dfba8ce1c60a5)) - Julian Hoever
- remove InProjectVHDLTemplate and InMemoryVHDLTemplate - ([e625399](https://github.com/es-ude/elastic-ai.creator/commit/e6253997447b0976de4ed60ec671de80ec6740a6)) - Julian Hoever
- remove RawTemplate class - ([eb91cd8](https://github.com/es-ude/elastic-ai.creator/commit/eb91cd81475a6a9aa94fc8ab4ccf3457cef55d01)) - Julian Hoever
- remove deprecated and broken relu and tanh implementations - ([286686c](https://github.com/es-ude/elastic-ai.creator/commit/286686cd6a2a185a94c03585f41d15dea794b1a2)) - Julian Hoever

---
## [0.37.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.1..v0.37.2) - 2023-05-07

### Bug Fixes

- try manual publishing - ([c8b6c35](https://github.com/es-ude/elastic-ai.creator/commit/c8b6c355896c1f3b0630c227af8414f281b5d3ff)) - Julian Hoever

---
## [0.37.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.37.0..v0.37.1) - 2023-05-07

### Bug Fixes

- try to fix semantic release - ([2625e89](https://github.com/es-ude/elastic-ai.creator/commit/2625e8982c021cbf5b778e95194cc53170ab0afb)) - Julian Hoever

---
## [0.37.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.36.0..v0.37.0) - 2023-05-05

### Features

- assert that all inserted variables exists in template and remove AbstractBaseTemplate - ([51f1a08](https://github.com/es-ude/elastic-ai.creator/commit/51f1a0883a8d0a54caee66080ef85f84049ad806)) - Julian Hoever

### Miscellaneous Chores

- add  force-publish workflow - ([b59268d](https://github.com/es-ude/elastic-ai.creator/commit/b59268d15b8ef605c6dbb48e606f5b1ad746548f)) - Lukas Einhaus
- update force publish workflow - ([9a0a7ac](https://github.com/es-ude/elastic-ai.creator/commit/9a0a7aca438f92e728c0310ec16adb0ded902f29)) - Lukas Einhaus
- update force-publish workflow - ([c7b011c](https://github.com/es-ude/elastic-ai.creator/commit/c7b011cd289baa1615cde11224f2a0ec25221e15)) - Lukas Einhaus
- update force-publish workflow - ([a56d2a9](https://github.com/es-ude/elastic-ai.creator/commit/a56d2a986102c26b925f20e982dd6af1e5b2fdfc)) - Lukas Einhaus

### Refactoring

- **(unit)** remove duplicated test - ([cfd304e](https://github.com/es-ude/elastic-ai.creator/commit/cfd304e630ba4f13ee87fc074c7d05fd99b1c98a)) - Julian Hoever
- remove unused parameter - ([89ca654](https://github.com/es-ude/elastic-ai.creator/commit/89ca65467a983230a1dc54d8b1502e82185f2acc)) - Julian Hoever

---
## [0.36.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.35.0..v0.36.0) - 2023-04-26

### Bug Fixes

- **(unit)** fix that test ignores parameter - ([f448919](https://github.com/es-ude/elastic-ai.creator/commit/f448919ff4882696c0991d6aec3608616e258596)) - Julian Hoever
- add missing save_to function - ([ef24ee2](https://github.com/es-ude/elastic-ai.creator/commit/ef24ee21672099359867bc4a74f5804af0c10158)) - Julian Hoever
- fix syntax errors - ([f9b57e4](https://github.com/es-ude/elastic-ai.creator/commit/f9b57e4f8173dc0bd52c21b1da351304ceb5a122)) - Julian Hoever
- fix syntax error - ([396f5c4](https://github.com/es-ude/elastic-ai.creator/commit/396f5c45b382454d6cc97e4be573fcfe45a4592a)) - Julian Hoever
- correct expected connections - ([2fb0f8e](https://github.com/es-ude/elastic-ai.creator/commit/2fb0f8edc45a7a38e2a9b7433dee90f139b10006)) - Lukas Einhaus

### Features

- **(unit)** test all subdesigns generated by sequential layer gets a unique name - ([009405b](https://github.com/es-ude/elastic-ai.creator/commit/009405bc64cd5e8a86909330bb450ee58ee98289)) - Julian Hoever
- **(unit)** add tests for sequential model with two layer - ([df73a4f](https://github.com/es-ude/elastic-ai.creator/commit/df73a4fb27a8867a4b633c4ffdd737ead34d2f16)) - Julian Hoever
- introduce abstract Translatable class - ([5d9fa2d](https://github.com/es-ude/elastic-ai.creator/commit/5d9fa2d167a8c46c301bb4a0da25718b1fcf0dee)) - Julian Hoever
- sequential layer can have a name - ([9e46938](https://github.com/es-ude/elastic-ai.creator/commit/9e46938e9e5fc6960e70bef26aa72ec51566a007)) - Julian Hoever
- test signal definitions, layer connections and instantiations separately - ([65201c8](https://github.com/es-ude/elastic-ai.creator/commit/65201c83bae07c62efcd705f67f34d9ff88da557)) - Julian Hoever
- autogenerate sequential signal connections - ([6dfca07](https://github.com/es-ude/elastic-ai.creator/commit/6dfca078b735a3387b65c20de601426ea27371c6)) - Lukas Einhaus

### Miscellaneous Chores

- adjust main.yml - ([93550cc](https://github.com/es-ude/elastic-ai.creator/commit/93550cccd7eda401dc7f759da8efe048661c2573)) - Lukas Einhaus
- adjust main.yml - ([359889c](https://github.com/es-ude/elastic-ai.creator/commit/359889c28d4ff4776eec3ff5e6d22dfab450cb4e)) - Lukas Einhaus
- adjust main.yml - ([2680a3a](https://github.com/es-ude/elastic-ai.creator/commit/2680a3a142e0df535fd07b716fdd6f5d7b0c1c14)) - Lukas Einhaus

### Refactoring

- remove unused translatable protocol and rename module - ([9d59f8c](https://github.com/es-ude/elastic-ai.creator/commit/9d59f8cd533b32baf6f90365e0db5a8b18d1c5a7)) - Julian Hoever
- remove unused import - ([602c137](https://github.com/es-ude/elastic-ai.creator/commit/602c1376cefe7dc4a95ef7cf04b9f67b0e2cf1e3)) - Julian Hoever
- fix/add missing type annotations - ([d47a8c1](https://github.com/es-ude/elastic-ai.creator/commit/d47a8c1c8919066e557a702f3bccc3928f35fa69)) - Julian Hoever
- use identity instead of linear layer to simplify test - ([28a75c3](https://github.com/es-ude/elastic-ai.creator/commit/28a75c337734b6bed887b1a3f9fc0369d92d330b)) - Julian Hoever
- rename FPLinear1d to FPLinear - ([5550dd9](https://github.com/es-ude/elastic-ai.creator/commit/5550dd97956171f53edc59e534dd02161c463133)) - Julian Hoever
- reduce code duplication - ([ae65808](https://github.com/es-ude/elastic-ai.creator/commit/ae65808bc66ebd2982a80ec3b6c5d70f749723d8)) - Julian Hoever

---
## [0.35.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.34.0..v0.35.0) - 2023-04-17

### Bug Fixes

- **(unit)** fix tests and remove hard sigmoid test in sequential test case - ([a1ada6f](https://github.com/es-ude/elastic-ai.creator/commit/a1ada6f0ceec750bb80abf866d28f96719f2f1f9)) - Julian Hoever
- set correct resource options in rom and fix signal definitions - ([2c2964c](https://github.com/es-ude/elastic-ai.creator/commit/2c2964ceaa746163ebbeaef09181e09c06ecb4f2)) - Julian Hoever

### Features

- add translate_to_vhdl function - ([ba0edc2](https://github.com/es-ude/elastic-ai.creator/commit/ba0edc25b93075cbb2d104c2216dcc15df36c13c)) - Julian Hoever
- implement translatable identity module - ([54327fa](https://github.com/es-ude/elastic-ai.creator/commit/54327fa3e45ca3617d642134ca8d842e7d2afc4c)) - Julian Hoever
- add indentations to template - ([aa254d1](https://github.com/es-ude/elastic-ai.creator/commit/aa254d12f38712e798db9b31a5a58e197a44121a)) - Julian Hoever
- generate first base template - ([a65d72e](https://github.com/es-ude/elastic-ai.creator/commit/a65d72ea1ad2dd87a0443b56711d11ce321d14b6)) - Lukas Einhaus
- generate template from manifest.toml - ([51276a0](https://github.com/es-ude/elastic-ai.creator/commit/51276a01de5ff37bedc598f5c758e3dc681aa49c)) - Lukas Einhaus
- use fixed base template - ([432dfd9](https://github.com/es-ude/elastic-ai.creator/commit/432dfd9518a0a33a7ba08cf95436f9472b274b52)) - Lukas Einhaus

### Miscellaneous Chores

- move to python3.11 - ([389e4ec](https://github.com/es-ude/elastic-ai.creator/commit/389e4ec6d60dbf594026993bf8f7d94d4bea1da8)) - Lukas Einhaus
- upgrade to python3.11 - ([f39c779](https://github.com/es-ude/elastic-ai.creator/commit/f39c7798f4ccc3799c707c8dcefbd176f9b6813b)) - Lukas Einhaus

### Refactoring

- remove unused imports - ([d9592ec](https://github.com/es-ude/elastic-ai.creator/commit/d9592ecb3677ba8050cb737bbc112987e72f25b5)) - Julian Hoever
- remove unused imports - ([735dcfa](https://github.com/es-ude/elastic-ai.creator/commit/735dcfaaba2ed4cace1b30d328fdaaf5433c5c42)) - Julian Hoever
- remove superfluous module protocols - ([4e25dc6](https://github.com/es-ude/elastic-ai.creator/commit/4e25dc65dfa0c226c298f5e589a6c887d72a3c19)) - Lukas Einhaus

### Wip

- set unique names for designs - ([59fbe4c](https://github.com/es-ude/elastic-ai.creator/commit/59fbe4c0bffac836718084f8acb8b04c7bb3d9c8)) - Julian Hoever
- start specifying meta files for designs/layers - ([b8585e2](https://github.com/es-ude/elastic-ai.creator/commit/b8585e2cecb501927ba52c28b57ded85ebfdc52c)) - Lukas Einhaus
- start pass_through for base template signals - ([6d47cbb](https://github.com/es-ude/elastic-ai.creator/commit/6d47cbbabd2da358b98366a5ad668492541d407c)) - Lukas Einhaus
- try pytest-bdd - ([0d642d9](https://github.com/es-ude/elastic-ai.creator/commit/0d642d912517a425b7ca2c0617a8ac8d0305303a)) - Lukas Einhaus

---
## [0.34.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.3..v0.34.0) - 2023-04-06

### Bug Fixes

- correct import paths - ([169f868](https://github.com/es-ude/elastic-ai.creator/commit/169f8686108845702f01482170df53e3fabbfe8b)) - Lukas Einhaus

### Features

- binary_arithmetics - ([54e38d5](https://github.com/es-ude/elastic-ai.creator/commit/54e38d57f27db2d8d0baff5fee3c35a91e26ecd9)) - Lukas Einhaus
- make precomputed scalar functions use unified interface - ([6b59da5](https://github.com/es-ude/elastic-ai.creator/commit/6b59da53a896db7676119de2f74129bcc47287ed)) - Lukas Einhaus

### Miscellaneous Chores

- remove unneeded import - ([e3df52a](https://github.com/es-ude/elastic-ai.creator/commit/e3df52a091e4673460f7b1ad733d766bad4afd02)) - Lukas Einhaus
- add mypy and pylint to pyproject.toml - ([aad5549](https://github.com/es-ude/elastic-ai.creator/commit/aad5549c7bbfbaf648fc3bbab0f77cd6c0ad49ca)) - Lukas Einhaus

---
## [0.33.3](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.2..v0.33.3) - 2023-04-06

### Bug Fixes

- set correct rom names - ([9570826](https://github.com/es-ude/elastic-ai.creator/commit/95708269900ca99b79da9ba37078f593724e5d17)) - Julian Hoever
- remove DualPort2ClockRam design - ([f9224c6](https://github.com/es-ude/elastic-ai.creator/commit/f9224c6809b3a6f72bfe0405419de494b099b17c)) - Julian Hoever

### Documentation

- remove deprecated documentation - ([11b9945](https://github.com/es-ude/elastic-ai.creator/commit/11b9945bf3b6bf96899a09751963a93eb98d846d)) - Julian Hoever

### Refactoring

- rename nn to base_modules - ([44207a8](https://github.com/es-ude/elastic-ai.creator/commit/44207a8f72e426fcd1cb4acc5b3c53c4ac8fa2f2)) - Julian Hoever
- rename translatable_modules to nn - ([333ac57](https://github.com/es-ude/elastic-ai.creator/commit/333ac5776788367ed3a8c17632fa20e11556f43e)) - Julian Hoever
- move hardware specific lstm parts to nn package - ([bfe575c](https://github.com/es-ude/elastic-ai.creator/commit/bfe575c50291388eb2f8b243d3411ff9e847490c)) - Julian Hoever
- reorder class definitions to avoid the usage of quotes - ([780c1fe](https://github.com/es-ude/elastic-ai.creator/commit/780c1fe67d18893400226e8acc6e77504da6a6ad)) - Julian Hoever
- move lstm designs in designs directory - ([36a807b](https://github.com/es-ude/elastic-ai.creator/commit/36a807b00794bac42a5018759e2ec09238bf043e)) - Julian Hoever

### Wip

- move vhdl implementation specific parts in the nn package - ([17d13a1](https://github.com/es-ude/elastic-ai.creator/commit/17d13a1d3b44e53ffd9debf2a8d8c85172236098)) - Julian Hoever
- start adding DualPort2ClockRam design - ([e9a1b55](https://github.com/es-ude/elastic-ai.creator/commit/e9a1b559f68fe25b627262442ad08566419455fc)) - Julian Hoever

---
## [0.33.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.1..v0.33.2) - 2023-03-23

### Bug Fixes

- **(translation)** add missing ROMs and set correct names in fp_linear1d template - ([ad4c6f0](https://github.com/es-ude/elastic-ai.creator/commit/ad4c6f095102965ff1dffa83dab4f2cb9749ce49)) - Julian Hoever
- **(translation)** add missing rom files and calculate correct twos complement - ([f700409](https://github.com/es-ude/elastic-ai.creator/commit/f70040956b7637844a471a5eff171d9cc6ba4c72)) - Julian Hoever
- **(unit)** fix failing unittests that are using the linear1d layer and design - ([ff582e1](https://github.com/es-ude/elastic-ai.creator/commit/ff582e185ea01cc6282cb4553e14701e88a9d8f8)) - Julian Hoever
- fix type annotation - ([8da1107](https://github.com/es-ude/elastic-ai.creator/commit/8da1107b2640d695816c71dd3980c0783b522122)) - Julian Hoever
- small import fix - ([07d2e29](https://github.com/es-ude/elastic-ai.creator/commit/07d2e29c36e60d35066d2145782223aa42d64519)) - Julian Hoever

### Miscellaneous Chores

- allow all torch versions >= 1.11 and < 2.0 - ([7321d7c](https://github.com/es-ude/elastic-ai.creator/commit/7321d7cf5694588a607975d13958edbfa5a3b331)) - Julian Hoever

### Refactoring

- small file and folder renames - ([9602a86](https://github.com/es-ude/elastic-ai.creator/commit/9602a868e6067889e2386c764e173c36f33e304c)) - Lukas Einhaus

### Wip

- add rom to linear design - ([35cea06](https://github.com/es-ude/elastic-ai.creator/commit/35cea06b1319fac138ea82b75104b3a6e141c574)) - Lukas Einhaus

---
## [0.33.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.33.0..v0.33.1) - 2023-03-15

### Bug Fixes

- wrong fixed point config object used for linear layers - ([3626113](https://github.com/es-ude/elastic-ai.creator/commit/36261136add4b4d378598dc8c9e858240f6557c5)) - Lukas Einhaus
- usage of lstm output in lstm_network impl - ([2e16141](https://github.com/es-ude/elastic-ai.creator/commit/2e1614184cdaa073fdcc686b891748861fe5c7cc)) - Lukas Einhaus

---
## [0.33.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.32.1..v0.33.0) - 2023-03-15

### Bug Fixes

- correctly pad rom memory - ([fe768d5](https://github.com/es-ude/elastic-ai.creator/commit/fe768d5f93c34ade65c24479c70f3528c66b0408)) - Lukas Einhaus

### Features

- add rom design for saving weights - ([75862b7](https://github.com/es-ude/elastic-ai.creator/commit/75862b7db4e64173daf7e6cdcb8413b0f510d396)) - Lukas Einhaus

### Refactoring

- rom design - ([975ad7e](https://github.com/es-ude/elastic-ai.creator/commit/975ad7e139a15466338cff72cfedeedf0c532f75)) - Lukas Einhaus
- use rom design in implementation - ([a8bfe4a](https://github.com/es-ude/elastic-ai.creator/commit/a8bfe4a2395a9bd81aa33f1989154f84a21bf001)) - Lukas Einhaus
- move conversions to twos complement from designs to translatable modules - ([50ada18](https://github.com/es-ude/elastic-ai.creator/commit/50ada185de5a081295515e16773b7fefdaa107eb)) - Lukas Einhaus

---
## [0.32.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.32.0..v0.32.1) - 2023-03-14

### Bug Fixes

- set library for lstm_cell - ([2b3a565](https://github.com/es-ude/elastic-ai.creator/commit/2b3a565039672ca89a1c5f593db5a5f32742f771)) - Lukas Einhaus
- typo in test for lstm cell designs - ([2ffeaec](https://github.com/es-ude/elastic-ai.creator/commit/2ffeaecf3ba7c3c0946c57ab3bee92af55746887)) - Lukas Einhaus

---
## [0.32.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.31.0..v0.32.0) - 2023-03-14

### Bug Fixes

- **(translation)** correct values for x/y_address_width - ([c7af1af](https://github.com/es-ude/elastic-ai.creator/commit/c7af1af71ef9319ed2ee7fffd7afcbaa5ffda580)) - Lukas Einhaus
- tests and remove type annotations leading to deps - ([75ed6cc](https://github.com/es-ude/elastic-ai.creator/commit/75ed6cc4f3a92b80656433b8209c0c932595900e)) - Lukas Einhaus

### Features

- **(translation)** sequential layer with bufferless layers - ([d7cea69](https://github.com/es-ude/elastic-ai.creator/commit/d7cea69ad0696f63e00762991e7407ad09d8a94c)) - Lukas Einhaus
- **(translation)** add support for single buffered module to sequential - ([5402782](https://github.com/es-ude/elastic-ai.creator/commit/5402782c0c37a6838b77b19d8040d256217d72ba)) - Lukas Einhaus
- add linear layer to lstm network - ([48982f0](https://github.com/es-ude/elastic-ai.creator/commit/48982f0aca675098b77edb2c8419b09ebc388835)) - Lukas Einhaus
- add linear layer to lstm network - ([bccb50c](https://github.com/es-ude/elastic-ai.creator/commit/bccb50cd6e3bc4e3e3115a41e051a1b962f6be52)) - Lukas Einhaus

### Miscellaneous Chores

- update gh workflow to match new tests location - ([58b7151](https://github.com/es-ude/elastic-ai.creator/commit/58b71513d05aa0bbf34533dc72b070ceaee34e83)) - Lukas Einhaus
- update gh-workflow - ([b1d714d](https://github.com/es-ude/elastic-ai.creator/commit/b1d714d4d408917ddd389db7fa29eed6c0230684)) - Lukas Einhaus
- update gh-workflow - ([7418a7b](https://github.com/es-ude/elastic-ai.creator/commit/7418a7b46764c808649a78f7e132a8fe51880376)) - Lukas Einhaus

### Refactoring

- **(nn)** replace fixed point factory by fixed point config - ([b5a08ac](https://github.com/es-ude/elastic-ai.creator/commit/b5a08acc11453ad550e2457836f1f4a2f5cbbae1)) - Lukas Einhaus
- **(translation)** move modules - ([24e522f](https://github.com/es-ude/elastic-ai.creator/commit/24e522fb10224bbd4065d841b2df97fa0f561021)) - Lukas Einhaus
- **(translation)** refactor autowiring for sequential network module - ([431862f](https://github.com/es-ude/elastic-ai.creator/commit/431862f21b6f074021973a88789a654461ae269e)) - Lukas Einhaus
- start moving relevant tests to top-level tests dir - ([577f43d](https://github.com/es-ude/elastic-ai.creator/commit/577f43d16a30fb1e6cc73c7dca7a4d6391559f79)) - Lukas Einhaus
- tweak module hierarchy - ([40bc371](https://github.com/es-ude/elastic-ai.creator/commit/40bc371d6602c504ed6e69542ef3a51d525fda70)) - Lukas Einhaus
- remove code generation dependency on fixed point data types - ([4d83d1b](https://github.com/es-ude/elastic-ai.creator/commit/4d83d1bc8f1a91de6dfd8995373155151d74fc25)) - Lukas Einhaus
- lstm roms - ([a2e08ec](https://github.com/es-ude/elastic-ai.creator/commit/a2e08ec2f1492cd0efc9f4e60b76b4a42c0d093f)) - Lukas Einhaus

### Wip

- overhaul of hw block generation - ([475136f](https://github.com/es-ude/elastic-ai.creator/commit/475136f03849ea7001a3e00be14bb8615809d462)) - Lukas Einhaus
- code generation design - ([a1bad85](https://github.com/es-ude/elastic-ai.creator/commit/a1bad8513470c3f54ff4b455257ff2fc27177565)) - Lukas Einhaus
- generate rom files for lstm - ([68088cd](https://github.com/es-ude/elastic-ai.creator/commit/68088cd960b8a947d35bf1ecaae93e2708e06527)) - Lukas Einhaus
- lstm roms, hardcoded zeros - ([a1f7363](https://github.com/es-ude/elastic-ai.creator/commit/a1f73634e5ca2c98f1109350c2fef949280cf265)) - Lukas Einhaus

---
## [0.31.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.4..v0.31.0) - 2023-02-22

### Bug Fixes

- fix incorrect vector signal initialization - ([3c68255](https://github.com/es-ude/elastic-ai.creator/commit/3c68255057dad325ab4ba89601f6f1e2384f0d95)) - Lukas Einhaus
- fix incorrect vector signal initialization - ([3b23f7a](https://github.com/es-ude/elastic-ai.creator/commit/3b23f7a64afda8cd2ee320f6af7dc372f9daf5e2)) - Lukas Einhaus
- type annotations for tracing module - ([da598a9](https://github.com/es-ude/elastic-ai.creator/commit/da598a92fc8f76b3c19d0b960d77122b82d171ac)) - Lukas Einhaus
- fix unit tests after major rebase - ([3b596e9](https://github.com/es-ude/elastic-ai.creator/commit/3b596e9c20e302bbf42efda7577e01498c05bc6c)) - Lukas Einhaus
- typing - ([b0bfa39](https://github.com/es-ude/elastic-ai.creator/commit/b0bfa39b98555b37f0d2626a235ac74987e2c9ad)) - Lukas Einhaus

### Features

- **(translation)** add missing suffixes - ([cb05d0f](https://github.com/es-ude/elastic-ai.creator/commit/cb05d0f3f8665ac98c0cff70cbb2dbd8d2a5b2f2)) - Julian Hoever
- add connectable base in/out signals - ([7ad67f9](https://github.com/es-ude/elastic-ai.creator/commit/7ad67f916815b692daddae98d4c93b9a5eb21641)) - Lukas Einhaus
- add logic and logic vector signals - ([1947baa](https://github.com/es-ude/elastic-ai.creator/commit/1947baac032e1b3958344779a00b84615b5581a1)) - Lukas Einhaus
- introduce vhdl_design class - ([20566f6](https://github.com/es-ude/elastic-ai.creator/commit/20566f600383ccb68fed60483bede9db5436913f)) - Lukas Einhaus
- add connectable base in/out signals - ([fea05ed](https://github.com/es-ude/elastic-ai.creator/commit/fea05ed0507550c23701bf6f5e3a562b68af73d4)) - Lukas Einhaus
- add logic and logic vector signals - ([551f241](https://github.com/es-ude/elastic-ai.creator/commit/551f24113be03b45ab1811cb734e521671620d89)) - Lukas Einhaus
- introduce vhdl_design class - ([2431ba4](https://github.com/es-ude/elastic-ai.creator/commit/2431ba40b71c19dff161ea9b78d7b5277970a6f9)) - Lukas Einhaus
- add data flow node, sink node and source node - ([9a511de](https://github.com/es-ude/elastic-ai.creator/commit/9a511de4d2618c3131abcd3c481b918ffa96545e)) - Lukas Einhaus

### Miscellaneous Chores

- introduce private package import lint rule - ([b497e1c](https://github.com/es-ude/elastic-ai.creator/commit/b497e1ca3c512d2414cc0736305e19a867251741)) - Lukas Einhaus
- tweak import contract - ([306de20](https://github.com/es-ude/elastic-ai.creator/commit/306de20163ad6e751b5e8d5e66601e90d1856b50)) - Lukas Einhaus
- update deps - ([00700fe](https://github.com/es-ude/elastic-ai.creator/commit/00700fe92b86442cc7e0db29794fa78d20ba48f9)) - Lukas Einhaus
- add class diagram for vhdldesign - ([01c63e0](https://github.com/es-ude/elastic-ai.creator/commit/01c63e02759ca71c93dc3f985d416d3ffa2c31af)) - Lukas Einhaus
- clean up external deps - ([d1be65a](https://github.com/es-ude/elastic-ai.creator/commit/d1be65aee7144be24c79a280c93537115acd2e31)) - Lukas Einhaus

### Refactoring

- merge utilities for testing code - ([333c09a](https://github.com/es-ude/elastic-ai.creator/commit/333c09a9b396f450e24d7d2390daa8b502b5cdac)) - Lukas Einhaus
- move file reading to CodeTestCase - ([3cc9c5e](https://github.com/es-ude/elastic-ai.creator/commit/3cc9c5e4c67fea3e8bea566eeb1a30feea7c1b56)) - Lukas Einhaus
- remove unintended print statement - ([b43befd](https://github.com/es-ude/elastic-ai.creator/commit/b43befdb529389a8cc8c08d087631ca45163f51c)) - Lukas Einhaus
- move code test utility files - ([d390af1](https://github.com/es-ude/elastic-ai.creator/commit/d390af12f9658952fd08b4493b467ee820c45f5f)) - Lukas Einhaus
- rename test_logic_signals - ([f817425](https://github.com/es-ude/elastic-ai.creator/commit/f817425f96895cdf52ff184f7cc32473e3c85fe9)) - Lukas Einhaus
- simplify architecture - ([1f5f1f1](https://github.com/es-ude/elastic-ai.creator/commit/1f5f1f19510f6dd9282e5bdda5beab904b2328b3)) - Lukas Einhaus
- merge utilities for testing code - ([6704e87](https://github.com/es-ude/elastic-ai.creator/commit/6704e87c1964615aa8b5d24042703a29b0b9ca1f)) - Lukas Einhaus
- move file reading to CodeTestCase - ([2c0ecb4](https://github.com/es-ude/elastic-ai.creator/commit/2c0ecb44ab3663e26e178bbb650b8c4f5298b195)) - Lukas Einhaus
- remove unintended print statement - ([5f2891e](https://github.com/es-ude/elastic-ai.creator/commit/5f2891e5c02d0448b357a6aa8b6433d2da25f4bf)) - Lukas Einhaus
- move code test utility files - ([8efc21e](https://github.com/es-ude/elastic-ai.creator/commit/8efc21e0df17149a4f36f134d940f2bc98cf1c44)) - Lukas Einhaus
- rename test_logic_signals - ([9d16019](https://github.com/es-ude/elastic-ai.creator/commit/9d160195fd3c03ddaffb0b5609e9b1d5dcc56d02)) - Lukas Einhaus
- remove obsolete graph package - ([ac53d76](https://github.com/es-ude/elastic-ai.creator/commit/ac53d7684135e3bab4d940d1c80951b297d19d77)) - Lukas Einhaus
- remove obsolete vhdl_design module - ([d4e61bd](https://github.com/es-ude/elastic-ai.creator/commit/d4e61bd7440d42a878f7539af7c256d637c2b7ba)) - Lukas Einhaus
- simplify signals and move classes - ([aacb702](https://github.com/es-ude/elastic-ai.creator/commit/aacb7021bcb83cb96053092640a7b7cdc6e2077d)) - Lukas Einhaus
- use relative imports inside packages - ([ef8d588](https://github.com/es-ude/elastic-ai.creator/commit/ef8d58878058b2eb6ef5f177171350c6759132f7)) - Lukas Einhaus
- simplify data flow node - ([82c8ba8](https://github.com/es-ude/elastic-ai.creator/commit/82c8ba825bfa3b5d367bc3d6f473d2055ef217d6)) - Lukas Einhaus
- remove/move/merge protocols - ([8391a1c](https://github.com/es-ude/elastic-ai.creator/commit/8391a1c7e459bbf176840976a741317a28f3abd6)) - Lukas Einhaus
- only return file object from package without opening it - ([2c57287](https://github.com/es-ude/elastic-ai.creator/commit/2c572879a98a4af72978bbd471704395606b96fc)) - Lukas Einhaus
- separate template from file - ([73f00e0](https://github.com/es-ude/elastic-ai.creator/commit/73f00e0e2e1e6302f2d8325fe9075d9bd51c25a3)) - Lukas Einhaus
- remove deprecated vhdl.language module - ([e29f6da](https://github.com/es-ude/elastic-ai.creator/commit/e29f6da7e76018dce7d32f9698a7973de6e5e832)) - Lukas Einhaus
- move modules/classes to fix dependency issues - ([22564d7](https://github.com/es-ude/elastic-ai.creator/commit/22564d7ce4b05770d49078c0d5ce13fe3ace231d)) - Lukas Einhaus
- move more modules/classes to fix dependency issues - ([ae82c14](https://github.com/es-ude/elastic-ai.creator/commit/ae82c143100ddb9a49a7cfae36d8ea5289789fa4)) - Lukas Einhaus
- move more modules/classes to fix dependency issues - ([0e25d94](https://github.com/es-ude/elastic-ai.creator/commit/0e25d949c94a6efa6e0ffe6f0530f09e72c2f5b5)) - Lukas Einhaus
- adjust architecture in design.md and move modules accordingly - ([236e6c3](https://github.com/es-ude/elastic-ai.creator/commit/236e6c3457cbbb413b8fd79015bfe1e97c49563d)) - Lukas Einhaus
- simplify signals - ([884ad64](https://github.com/es-ude/elastic-ai.creator/commit/884ad648fde4381a4dd892542bf576a7cd2d090b)) - Lukas Einhaus
- simplify ports - ([4bdf84a](https://github.com/es-ude/elastic-ai.creator/commit/4bdf84a4f72f1b99d89afa84de234c74a637fcd0)) - Lukas Einhaus
- remove superfluous protocol - ([741c53b](https://github.com/es-ude/elastic-ai.creator/commit/741c53baf3ca0ee9ccb27d5cf5a64d172eac7781)) - Lukas Einhaus

### Wip

- start design for signal connections - ([510c1c7](https://github.com/es-ude/elastic-ai.creator/commit/510c1c71bc37aca22165c0c895ec850b416c797b)) - Lukas Einhaus
- start design for signal connections - ([e6d451e](https://github.com/es-ude/elastic-ai.creator/commit/e6d451eb38b4bf2ff700569bb594d40f26b52407)) - Lukas Einhaus
- add dataflow node - ([4db8583](https://github.com/es-ude/elastic-ai.creator/commit/4db8583379a1429e4e181e503310accfc5b5d962)) - Lukas Einhaus
- simplify number_representations.py - ([9e84997](https://github.com/es-ude/elastic-ai.creator/commit/9e84997a52bbeceb7b99be74f557df4b60b17806)) - Lukas Einhaus
- overhaul of hw block generation - ([5c878a2](https://github.com/es-ude/elastic-ai.creator/commit/5c878a22a60e77a6338430f325894555a81788ff)) - Lukas Einhaus

---
## [0.30.4](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.3..v0.30.4) - 2023-02-16

### Bug Fixes

- **(translation)** get rid of the duplicated suffix on rom component - ([9cd0e0b](https://github.com/es-ude/elastic-ai.creator/commit/9cd0e0be9481a286820eea5c8d5bdc9d28fcc0d8)) - Chao Qian

---
## [0.30.3](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.2..v0.30.3) - 2023-02-16

### Bug Fixes

- **(template)** linear layer template - ([96bdf03](https://github.com/es-ude/elastic-ai.creator/commit/96bdf030ca4c27d67a4978e3b8609ef57c40a01e)) - Chao Qian
- **(unit)** add rounding to prevent tests from failing due to floating point loss - ([b7314b7](https://github.com/es-ude/elastic-ai.creator/commit/b7314b797ef39c2f693554821ec7bb3d96689661)) - Julian Hoever

---
## [0.30.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.1..v0.30.2) - 2023-02-15

### Bug Fixes

- ignore single import mypy error - ([dd85159](https://github.com/es-ude/elastic-ai.creator/commit/dd851590719ec76ab66dc9d908493991fc235e7e)) - Julian Hoever
- use non-static path to example folder - ([613a152](https://github.com/es-ude/elastic-ai.creator/commit/613a152e65fbe0f7116a1f772fea8a3836d888af)) - Julian Hoever

### Miscellaneous Chores

- remove unused dependencies and update poetry lock - ([7b4b658](https://github.com/es-ude/elastic-ai.creator/commit/7b4b658c2649500809ade7efd716e8dca4153576)) - Julian Hoever

### Refactoring

- **(integration)** remove non-deterministic test - ([ebed2a7](https://github.com/es-ude/elastic-ai.creator/commit/ebed2a73beaba1f9e6abdc843eb5771cc1d34061)) - Julian Hoever
- **(nn)** remove unused module - ([d2e643b](https://github.com/es-ude/elastic-ai.creator/commit/d2e643b1368a5776829a0353730afa5039c19590)) - Julian Hoever
- **(unit)** move test in the unit folder - ([89df933](https://github.com/es-ude/elastic-ai.creator/commit/89df933b50eb35e0528042f81a37a59ba8630ff5)) - Julian Hoever
- **(unit)** move tensor_test_case in tests directory - ([3cf635b](https://github.com/es-ude/elastic-ai.creator/commit/3cf635b2d5ecbad524cfed75d4d4b7543c2dbcc2)) - Julian Hoever
- remove deprecated examples - ([eec3f0e](https://github.com/es-ude/elastic-ai.creator/commit/eec3f0e75a7875a8a2d1da9c2ffe586a4a18ebf9)) - Julian Hoever
- create integration test from POS tagger example - ([cb73343](https://github.com/es-ude/elastic-ai.creator/commit/cb73343957c6b75df2a741b08c66c11545b86f2d)) - Julian Hoever
- remove deprecated example - ([008241c](https://github.com/es-ude/elastic-ai.creator/commit/008241c8d5414cbe9478e1cdb226c22c48b2c663)) - Julian Hoever
- delete not relevant example - ([3c0fce9](https://github.com/es-ude/elastic-ai.creator/commit/3c0fce95db8c078b8e37e34d0018872164402c4f)) - Julian Hoever
- rename example - ([84d4792](https://github.com/es-ude/elastic-ai.creator/commit/84d479296c1930f4e7f334ae1d2fd89ba84b595a)) - Julian Hoever

### Wip

- add test to verify that the examples are executable - ([d71f875](https://github.com/es-ude/elastic-ai.creator/commit/d71f87569bc7d3950b9ed2da05a143896af39342)) - Julian Hoever

---
## [0.30.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.30.0..v0.30.1) - 2023-02-04

### Bug Fixes

- **(unit)** make test more deterministic - ([97fd410](https://github.com/es-ude/elastic-ai.creator/commit/97fd4101af93cf17d446cb0cb38a419080d5bee6)) - Julian Hoever

### Miscellaneous Chores

- remove vhdl scope - ([5c9571b](https://github.com/es-ude/elastic-ai.creator/commit/5c9571b384588551c7439f3e45ad63d8f718b79f)) - Julian Hoever

---
## [0.30.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.29.0..v0.30.0) - 2023-02-04

### Bug Fixes

- **(nn)** fix imports and use new FixedPointFactory features - ([e8c74c3](https://github.com/es-ude/elastic-ai.creator/commit/e8c74c34ec1c5a4b5189d74f2a19a993a5ae9779)) - Julian Hoever
- **(nn)** fix LSTMCell raises Error for unbatched input data and add a test for this case - ([5ce3e21](https://github.com/es-ude/elastic-ai.creator/commit/5ce3e2125b4bcd1115d77ebe5c833e52d58bad77)) - Julian Hoever
- **(translation)** add similar concept of translation arguments to fix the translation process - ([e387ae2](https://github.com/es-ude/elastic-ai.creator/commit/e387ae26918fbe8e4a0ee01ccc4361849746bd66)) - Julian Hoever
- **(translation)** change torch LSTM layer to our FixedPointLSTM layer - ([5e7a39a](https://github.com/es-ude/elastic-ai.creator/commit/5e7a39a78684c09a1d374476f8fb611019ae994f)) - Julian Hoever
- **(translation)** infer fixed_point_factory of linear and lstm in build functions - ([81df686](https://github.com/es-ude/elastic-ai.creator/commit/81df686fe13db5f85c91b65c73713b7da8e6c64f)) - Julian Hoever
- **(translation)** rename to .tpl.vhd - ([fe3c85c](https://github.com/es-ude/elastic-ai.creator/commit/fe3c85cd77d0f2fefb90f2d3ff6eadde8570d000)) - Julian Hoever
- **(translation)** remove sigmoid_resolution - ([dd4f033](https://github.com/es-ude/elastic-ai.creator/commit/dd4f03366920f1a3774772a16a49efaa8756d249)) - Julian Hoever
- **(translation)** use model.children() instead of model.modules() to avoid recursion - ([a3c349b](https://github.com/es-ude/elastic-ai.creator/commit/a3c349b13af0fef383b494850973d8ff9ac2dd68)) - Julian Hoever
- **(translation)** change not existing layer_id field to layer_name - ([f7425c5](https://github.com/es-ude/elastic-ai.creator/commit/f7425c515395243962db1517116b9961b1668cd7)) - Julian Hoever
- **(translation)** add layer_name to all vhdl templates and components - ([2d9c47d](https://github.com/es-ude/elastic-ai.creator/commit/2d9c47dc60642d94efeb58cc3014f6a7790a6f26)) - Julian Hoever
- **(translation)** fix errors in the lstm template and remove lstm_common component - ([c4a28ce](https://github.com/es-ude/elastic-ai.creator/commit/c4a28ce2f40dc84e7a5e4470c62a40911b73901f)) - Julian Hoever
- **(unit)** fix unit and integration tests to use the new layers correctly - ([0553017](https://github.com/es-ude/elastic-ai.creator/commit/05530178cf7fb64dc88cab82b89c24b2a1406e8d)) - Julian Hoever
- **(unit)** remove unused OperationType type and FakeQuant class - ([596dbd8](https://github.com/es-ude/elastic-ai.creator/commit/596dbd8cdf3cde67eedea2779a35ff682c9ac9f7)) - Julian Hoever
- adapt basic qtorch example to recent changes of the creator - ([a17d900](https://github.com/es-ude/elastic-ai.creator/commit/a17d9006240a67da97b8a539620aa1974e07e942)) - Julian Hoever
- fix some mypy errors and remove unused imports - ([08e2362](https://github.com/es-ude/elastic-ai.creator/commit/08e2362fa32efd13e388140ad58c93b0e79229b3)) - Julian Hoever

### Documentation

- add commit types and scopes - ([e759fd3](https://github.com/es-ude/elastic-ai.creator/commit/e759fd38fb41d413ccf03617f84f87f6df9aeb12)) - Julian Hoever

### Features

- **(integration)** convert example translate_linear_model to automated integration test - ([5d92d0b](https://github.com/es-ude/elastic-ai.creator/commit/5d92d0b15d8c0a1d76f842fd7a8bbc591bd1cf18)) - Julian Hoever
- **(nn)** rename quant_typings module to quantization and implement FakeQuant - ([0e5f24a](https://github.com/es-ude/elastic-ai.creator/commit/0e5f24aeb9f43258f9e971ffa777c585faff05f0)) - Julian Hoever
- **(nn)** integrate arithmetics for the linear layer - ([a961558](https://github.com/es-ude/elastic-ai.creator/commit/a9615581159ba4b962fac8458d9b76de0a61d98f)) - Julian Hoever
- **(nn)** remove input_quant and param_quant and add quantize function to arithmetics - ([ee91e42](https://github.com/es-ude/elastic-ai.creator/commit/ee91e42801b0d1163a0d52130fc578477da60c74)) - Julian Hoever
- **(nn)** implement concept of arithmetics - ([e7ad504](https://github.com/es-ude/elastic-ai.creator/commit/e7ad50471e2ac7300e0db781bd37cbba1364a5e6)) - Julian Hoever
- **(nn)** remove quantized_forward function and adopt tests - ([c865c73](https://github.com/es-ude/elastic-ai.creator/commit/c865c73a53e89c40ecebc9c4b49ba6d5c14256c1)) - Julian Hoever
- **(translation)** lstm uses fp hard sigmoid - ([fd265ac](https://github.com/es-ude/elastic-ai.creator/commit/fd265ac3e1ef7f11e28236705e4a38760462bddc)) - Julian Hoever
- **(translation)** integrate hard tanh layer - ([eb74d3a](https://github.com/es-ude/elastic-ai.creator/commit/eb74d3a3671616db37ba8f554332ca1ddc33dffe)) - Julian Hoever
- **(unit)** add unit tests for the LSTMBase layer - ([589f803](https://github.com/es-ude/elastic-ai.creator/commit/589f803fd858b22985485d795f4441a9abf97742)) - Julian Hoever
- **(unit)** add unit tests for the fixed point quant/dequant autograd functions - ([f82431c](https://github.com/es-ude/elastic-ai.creator/commit/f82431c164b9536899d0cca9b391a057add8187a)) - Julian Hoever
- **(unit)** improve TensorTestCase class - ([d4273a6](https://github.com/es-ude/elastic-ai.creator/commit/d4273a60c169669ddba5f80636d1430b69c77d90)) - Julian Hoever
- convert example parametrize_convolution to automated integration test - ([3dde1c2](https://github.com/es-ude/elastic-ai.creator/commit/3dde1c250fa4ebb617bbd543c9b26cb320d430f7)) - Julian Hoever
- add example to demonstrate that the new kinds of layers are trainable - ([231e325](https://github.com/es-ude/elastic-ai.creator/commit/231e325815c469596c63259c5f345dc9afb0f3b7)) - Julian Hoever
- small example for translating combination of lstm and linear layer - ([12e7101](https://github.com/es-ude/elastic-ai.creator/commit/12e7101e8c62e8424bc2ed580cfbe645e8d33510)) - Julian Hoever

### Miscellaneous Chores

- relax commitlint rules - ([108e361](https://github.com/es-ude/elastic-ai.creator/commit/108e361f763f23843b72c5620cbebd0c171a9433)) - Julian Hoever

### Refactoring

- **(integration)** move integration test to more specific location - ([0115399](https://github.com/es-ude/elastic-ai.creator/commit/01153996ac556eb9a96f404e8efed2af5bbdf1dd)) - Julian Hoever
- **(nn)** remove default bias value from linear layer - ([8d55471](https://github.com/es-ude/elastic-ai.creator/commit/8d5547180a50f07ee259f37cd8cd89ffe496e421)) - Julian Hoever
- **(nn)** add more precise type annotation - ([0c47fe0](https://github.com/es-ude/elastic-ai.creator/commit/0c47fe0b485cb71662ef017b7c454b848baa0b4f)) - Julian Hoever
- **(translation)** remove unnecessary print statement - ([2f8a0a7](https://github.com/es-ude/elastic-ai.creator/commit/2f8a0a75b602d6d7621f310e33ccf0bf0d5c1e28)) - Julian Hoever
- **(translation)** remove outdated evaluators - ([8c0009a](https://github.com/es-ude/elastic-ai.creator/commit/8c0009ae54dfed9f24223ca01a6b146ee0c06f04)) - Julian Hoever
- **(translation)** add fixed_point_factory property to fp layers and remove FixedPointLSTMCell - ([9f0a5d3](https://github.com/es-ude/elastic-ai.creator/commit/9f0a5d3505dc05d53aaf9fa9fb1c607049c661fd)) - Julian Hoever
- **(unit)** move unit test to correct location - ([c03c362](https://github.com/es-ude/elastic-ai.creator/commit/c03c3621c6cdef58a44e1c3e279d025ebdf34aa6)) - Julian Hoever
- remove examples belonging to the removed precomputation package - ([4dc681b](https://github.com/es-ude/elastic-ai.creator/commit/4dc681b18207dd92d767c97df2c70e2fd3e6cd2e)) - Julian Hoever

### Style

- beautify 6209df2bbc3c693f1829ce8b93822fc84152f69b - ([423b081](https://github.com/es-ude/elastic-ai.creator/commit/423b081476868df0a7f90fbcaeec16203670551f)) - github-actions

### Wip

- **(nn)** start implementing fixed point lstm - ([f0f4e3b](https://github.com/es-ude/elastic-ai.creator/commit/f0f4e3b1030a31a1972c6d6e94c7fe7cd9224147)) - Julian Hoever
- **(nn)** start implementing arithmetics classes - ([889d1d4](https://github.com/es-ude/elastic-ai.creator/commit/889d1d406cb0b06273c8ca6a8d7637a8ac10af10)) - Julian Hoever
- **(nn)** add naive fixed point matmul implementation - ([f09f82f](https://github.com/es-ude/elastic-ai.creator/commit/f09f82fa045491ad5acf504729a2143bee45b146)) - Julian Hoever

---
## [0.29.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.28.0..v0.29.0) - 2022-12-16

### Features

- set pypi project api token - ([37ba8c9](https://github.com/es-ude/elastic-ai.creator/commit/37ba8c9794acc6b4bdf64087c98c61172446fcb6)) - Julian Hoever

### Miscellaneous Chores

- tighten commitlint rules - ([47a35da](https://github.com/es-ude/elastic-ai.creator/commit/47a35da220ba1c6081af11b0a6e7945978f2fe77)) - Julian Hoever

---
## [0.28.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.27.0..v0.28.0) - 2022-12-16

### Miscellaneous Chores

- use gh-action provided by python-semantic-release - ([0d0321e](https://github.com/es-ude/elastic-ai.creator/commit/0d0321e44455d40c3b04929df13cccfe7056c35c)) - Lukas Einhaus
- add noop to semantic-release and trigger on workflow call - ([ecdb463](https://github.com/es-ude/elastic-ai.creator/commit/ecdb463514c0e5b8b0d0d22818071c728e6997e2)) - Lukas Einhaus
- add github action for test environment setup - ([8a38722](https://github.com/es-ude/elastic-ai.creator/commit/8a3872210155601a900b8ac59757808974961999)) - Lukas Einhaus
- rename actions yml - ([7882524](https://github.com/es-ude/elastic-ai.creator/commit/78825240f3cd78863110f516574d915781f3a4c5)) - Lukas Einhaus
- add commit hash to action reference - ([459a4cc](https://github.com/es-ude/elastic-ai.creator/commit/459a4ccd2487762c67a1be86f2ae071dc89396e8)) - Lukas Einhaus
- fetch repo in job instead of action - ([05d8bd1](https://github.com/es-ude/elastic-ai.creator/commit/05d8bd14a7c287c90755ffb68f2c899d3d182ad2)) - Lukas Einhaus
- specify shell in gh-action - ([a5fb59e](https://github.com/es-ude/elastic-ai.creator/commit/a5fb59e35e8b557011559ba5d55b68a452574710)) - Lukas Einhaus
- create cache-dir in action - ([f0ecc17](https://github.com/es-ude/elastic-ai.creator/commit/f0ecc17eedd1e9acdc6c0d4baa713eee6a5e2495)) - Lukas Einhaus
- reorder poetry calls and cache setup for action - ([2a0fb0d](https://github.com/es-ude/elastic-ai.creator/commit/2a0fb0d65d5bf80746b65d1d5f29f63cc59f36f1)) - Lukas Einhaus
- add missing argument to poetry configuration - ([1567b0c](https://github.com/es-ude/elastic-ai.creator/commit/1567b0c7269f14a1454b206b959c2c33862fe239)) - Lukas Einhaus
- enable semantic release for main again - ([6c93920](https://github.com/es-ude/elastic-ai.creator/commit/6c939203995883b390a20bc98b098a252563c669)) - Lukas Einhaus
- temporary relax commitlint rules - ([437c3d7](https://github.com/es-ude/elastic-ai.creator/commit/437c3d7cec0487f5754ec357fb4d313343fd2cbc)) - Julian Hoever
- temporary relax commitlint rules - ([7b007dc](https://github.com/es-ude/elastic-ai.creator/commit/7b007dc81ac7f1420b5d08e3a77f51d087a17dcf)) - Julian Hoever

### Refactoring

- **(nn)** remove unused import - ([14d1d60](https://github.com/es-ude/elastic-ai.creator/commit/14d1d60bb7b56c2c6bdd00feb767a8248a09699c)) - Julian Hoever
- **(unit)** rename package from qat to nn - ([e211ae6](https://github.com/es-ude/elastic-ai.creator/commit/e211ae63d9ee7fdc2c0fad15a40730399fac7654)) - Julian Hoever
- **(unit)** rename _init_quantizable_convolution function - ([2b57dbc](https://github.com/es-ude/elastic-ai.creator/commit/2b57dbcaa02f202c7654d8d15b53c84a0210ee1f)) - Julian Hoever

### Revert

- "chore: add commit hash to action reference" - ([e42d010](https://github.com/es-ude/elastic-ai.creator/commit/e42d01029b403029334dc2ed1a3311631361f9fb)) - Lukas Einhaus

---
## [0.27.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.26.1..v0.27.0) - 2022-12-15

### Bug Fixes

- **(onnx)** remove unmaintained onnx support - ([dc773d3](https://github.com/es-ude/elastic-ai.creator/commit/dc773d39fe2c0ea5785e3fb0bf7a43f3bf83495f)) - Lukas Einhaus
- **(onnx)** remove unmaintained onnx support - ([c200394](https://github.com/es-ude/elastic-ai.creator/commit/c200394239ff58ee31e5273d5999d731fbe5daca)) - Lukas Einhaus
- **(qat)** fix circular dependency - ([1d5615b](https://github.com/es-ude/elastic-ai.creator/commit/1d5615bf81757bf16904eb75c33fead69a68dd43)) - Julian Hoever
- **(qat)** fix the problem of wrong shapes for the QLSTM layer - ([b75f478](https://github.com/es-ude/elastic-ai.creator/commit/b75f47804016a3dfdad3f8d2dd575f4252cac5ff)) - Julian Hoever
- **(qat)** fix error when passing flat input data to _QLSTMBase and batch_first set to True - ([29918d1](https://github.com/es-ude/elastic-ai.creator/commit/29918d11c508e3e91fe00a0e07988be0ed198b35)) - Julian Hoever
- **(vhdl)** remove obsolete vhdl formatter - ([83d81e3](https://github.com/es-ude/elastic-ai.creator/commit/83d81e348152e047482ccc45a2ccaf6173f772d9)) - Lukas Einhaus
- **(vhdl)** remove obsolete vhdl formatter - ([128ba6b](https://github.com/es-ude/elastic-ai.creator/commit/128ba6bdbecd8763f77cc6862373446f5418201e)) - Lukas Einhaus

### Documentation

- **(readme)** move tests and remove deprecated lines - ([4a074a8](https://github.com/es-ude/elastic-ai.creator/commit/4a074a87fb31df535d415c2ab6aede7e4d7d8949)) - Julian Hoever
- **(vhdl)** add doc to VHDLFile - ([5fcf78b](https://github.com/es-ude/elastic-ai.creator/commit/5fcf78b87edf75ff3e9e818b1511aef00ffbf46a)) - Lukas Einhaus
- **(vhdl)** add doc to VHDLFile - ([a13b511](https://github.com/es-ude/elastic-ai.creator/commit/a13b511aa1c6d33749ab303fd1ee2760f6253ff2)) - Lukas Einhaus

### Features

- **(examples)** update qlstm sine wave example to the correctly implemented QLSTM layer - ([dc62cd2](https://github.com/es-ude/elastic-ai.creator/commit/dc62cd2aa05067b164009301ab7c5e110797c503)) - Julian Hoever
- **(qat)** add constraint type - ([dc4c4e5](https://github.com/es-ude/elastic-ai.creator/commit/dc4c4e57a9615a9be6941ecc750d3838458ff919)) - Julian Hoever
- **(qat)** remove constraints - ([6b7b483](https://github.com/es-ude/elastic-ai.creator/commit/6b7b4835dc9f9f6b6fc83bc619727aa948c19161)) - Julian Hoever
- **(vhdl)** support generation of layer connections - ([1d43c42](https://github.com/es-ude/elastic-ai.creator/commit/1d43c4212ef54c5488df7e7dc3829df31a7e8484)) - Lukas Einhaus
- **(vhdl)** generate portmap output_address - ([c6a26a6](https://github.com/es-ude/elastic-ai.creator/commit/c6a26a61d98c90fa29b02e6619116e67a4a67ac5)) - Lukas Einhaus
- **(vhdl)** add hw equivalent module tracer - ([3f2c2c7](https://github.com/es-ude/elastic-ai.creator/commit/3f2c2c7acc5046131d420d513a4bb3d3981ac0c5)) - Lukas Einhaus
- **(vhdl)** tracer records reference to module for call_module nodes - ([20ed7da](https://github.com/es-ude/elastic-ai.creator/commit/20ed7dab9677e476925a8b1250cbbc2004d43246)) - Lukas Einhaus
- **(vhdl)** generate vhdl signal definitions - ([53408f6](https://github.com/es-ude/elastic-ai.creator/commit/53408f6cb9daa5c44931e880fda0712c2924b822)) - Lukas Einhaus
- **(vhdl)** generate vhdl signal definitions - ([c593d3d](https://github.com/es-ude/elastic-ai.creator/commit/c593d3d501082595d4918be3c3425b6d9c636332)) - Lukas Einhaus
- **(vhdl)** generate layer instantiations - ([7a75fc3](https://github.com/es-ude/elastic-ai.creator/commit/7a75fc31780a6173424ffdcf3129bc60d5a83e59)) - Lukas Einhaus
- **(vhdl)** introduce HWBlocks - ([ab03eaf](https://github.com/es-ude/elastic-ai.creator/commit/ab03eaf28c74483fcd9dbd78d247d39e248bdea1)) - Lukas Einhaus
- **(vhdl)** extend code file with parameters - ([4833f8b](https://github.com/es-ude/elastic-ai.creator/commit/4833f8b2d5553cf02d322b8485587612cd67a9e8)) - Lukas Einhaus
- **(vhdl)** implement HWBlocks interface for sigmoid,linear - ([0177373](https://github.com/es-ude/elastic-ai.creator/commit/0177373eeddfa9c32100777bbcd7a94765dc1122)) - Lukas Einhaus
- **(vhdl)** add module_nodes to graph decorator - ([6d0a612](https://github.com/es-ude/elastic-ai.creator/commit/6d0a61217b36b9db8e9df19210e5f0d3aeed4ef2)) - Lukas Einhaus
- **(vhdl)** introduce HWEquivalentGraph - ([844bb84](https://github.com/es-ude/elastic-ai.creator/commit/844bb84a2d36e50f3de7ae4b713d370011d3240e)) - Lukas Einhaus
- **(vhdl)** introduce HWBlockCollection - ([a80bda2](https://github.com/es-ude/elastic-ai.creator/commit/a80bda2d705992030b18649ff99f3a6ce75d7ef3)) - Lukas Einhaus
- **(vhdl)** distinguish x/y width - ([2f52100](https://github.com/es-ude/elastic-ai.creator/commit/2f52100d32502520ce66a240bae90dd48e070ebd)) - Lukas Einhaus
- **(vhdl)** support generation of layer connections - ([fdd3176](https://github.com/es-ude/elastic-ai.creator/commit/fdd3176ba5d4652718e76dfd74dc92167f86b4f4)) - Lukas Einhaus
- **(vhdl)** generate portmap output_address - ([33e66d9](https://github.com/es-ude/elastic-ai.creator/commit/33e66d99b5b8c0801c93e520463ea92c6392e2b8)) - Lukas Einhaus
- **(vhdl)** add hw equivalent module tracer - ([fcb2e10](https://github.com/es-ude/elastic-ai.creator/commit/fcb2e102f5409a2e1dc358ce26e4cba6110a7e24)) - Lukas Einhaus
- **(vhdl)** tracer records reference to module for call_module nodes - ([ea1f0ee](https://github.com/es-ude/elastic-ai.creator/commit/ea1f0ee893c11065bdf17086badd248b998d29de)) - Lukas Einhaus
- **(vhdl)** generate vhdl signal definitions - ([5da3986](https://github.com/es-ude/elastic-ai.creator/commit/5da3986472a65e7f15cbedd3cba473ad4d67dde9)) - Lukas Einhaus
- **(vhdl)** generate vhdl signal definitions - ([c76d03d](https://github.com/es-ude/elastic-ai.creator/commit/c76d03db443cffd831abee60a8546aa3547c5fe6)) - Lukas Einhaus
- **(vhdl)** generate layer instantiations - ([925b837](https://github.com/es-ude/elastic-ai.creator/commit/925b837d33120d4bd1abdd8cae812d89d4979a9a)) - Lukas Einhaus
- **(vhdl)** introduce HWBlocks - ([141148f](https://github.com/es-ude/elastic-ai.creator/commit/141148f13c40725755a1b02b24d8899e01ae9ced)) - Lukas Einhaus
- **(vhdl)** extend code file with parameters - ([2bdfca3](https://github.com/es-ude/elastic-ai.creator/commit/2bdfca352b05756bb911eafb2b702f6536561b26)) - Lukas Einhaus
- **(vhdl)** implement HWBlocks interface for sigmoid,linear - ([53e05c7](https://github.com/es-ude/elastic-ai.creator/commit/53e05c7b772f8576b4f221e610360dc52601d852)) - Lukas Einhaus
- **(vhdl)** add module_nodes to graph decorator - ([bee0438](https://github.com/es-ude/elastic-ai.creator/commit/bee0438fb9b35d666998f4f516a1469c729b5829)) - Lukas Einhaus
- **(vhdl)** introduce HWEquivalentGraph - ([f0bdd73](https://github.com/es-ude/elastic-ai.creator/commit/f0bdd73d6e6e6ed9c8306a7771443e4d13e874ce)) - Lukas Einhaus
- **(vhdl)** introduce HWBlockCollection - ([cdcb324](https://github.com/es-ude/elastic-ai.creator/commit/cdcb324abe3c69893a782df075b24d734f244a6c)) - Lukas Einhaus
- **(vhdl)** distinguish x/y width - ([73549f9](https://github.com/es-ude/elastic-ai.creator/commit/73549f94a0c582170e2f43baea4afcb4c9c20124)) - Lukas Einhaus

### Miscellaneous Chores

- **(gh-workflow)** set correct path to unit and integration tests - ([538eb2f](https://github.com/es-ude/elastic-ai.creator/commit/538eb2f036f24ea99135f0e66ad59c3738e60231)) - Julian Hoever
- **(gh-workflow)** remove superfluous line - ([71edbc4](https://github.com/es-ude/elastic-ai.creator/commit/71edbc4369a59c90c561ff3e8b335bd85ecbba7e)) - Julian Hoever
- more specific commitlint rules - ([bbb88e9](https://github.com/es-ude/elastic-ai.creator/commit/bbb88e9080ecd873209f99aa01473b9d57bd2012)) - Lukas Einhaus
- update poetry.lock - ([0f78c4b](https://github.com/es-ude/elastic-ai.creator/commit/0f78c4bfdddad038bd69b1a92f3b1fba4c5ab9f8)) - Lukas Einhaus
- don't install extras prior publishing - ([effa8c0](https://github.com/es-ude/elastic-ai.creator/commit/effa8c004a2d8356a96e3869763e85e58ee92924)) - Lukas Einhaus
- tweak pyproject and commitlint - ([addc521](https://github.com/es-ude/elastic-ai.creator/commit/addc521744804fb8a6deeadde8510bd9fe37d87b)) - Lukas Einhaus
- add style again to pyproject and commitlint - ([d7aaf28](https://github.com/es-ude/elastic-ai.creator/commit/d7aaf28042881c272f851e5402135d15a149ec42)) - Lukas Einhaus

### Refactoring

- **(qat)** move BatchNormedActivatedConv1d from layers module to blocks module - ([6269522](https://github.com/es-ude/elastic-ai.creator/commit/6269522bcdb978a905c84693e6c9fa4bdc32bfa7)) - Julian Hoever
- **(qat)** split LayersTest class into classes for each layer - ([55c12b3](https://github.com/es-ude/elastic-ai.creator/commit/55c12b36ce0b9807ffa4f5dd8344e3b8143f1212)) - Julian Hoever
- **(qat)** remove unused import - ([b6cf349](https://github.com/es-ude/elastic-ai.creator/commit/b6cf3494b36cca9d2fd732a24952423b68ad6c46)) - Julian Hoever
- **(qat)** remove unused code - ([43e5992](https://github.com/es-ude/elastic-ai.creator/commit/43e5992f1e48d078113bba7863c0ac5e3e967ada)) - Julian Hoever
- **(qat)** remove unused code and make Identity quantizer public - ([dcd726e](https://github.com/es-ude/elastic-ai.creator/commit/dcd726e183c5b74b05c27155ec64cc08f395802e)) - Julian Hoever
- **(qat)** remove unused code - ([d49ca79](https://github.com/es-ude/elastic-ai.creator/commit/d49ca79cc4ec5834222a6253d96ff5402f905151)) - Julian Hoever
- **(qat)** remove noise comments and remove default quantizer from QLSTM and QLSTMCell layer - ([4a57ca9](https://github.com/es-ude/elastic-ai.creator/commit/4a57ca900a6c5dad1710f6d558c1ade17527d2b4)) - Julian Hoever
- **(qat)** remove default quantizer from QLSTM and QLSTMCell layer - ([cce2f8f](https://github.com/es-ude/elastic-ai.creator/commit/cce2f8f1d22c384f136583f74d3a2b396500b0e0)) - Julian Hoever
- **(qat)** create a _QLSTMCellBase and _QLSTMBase class to avoid default parameters - ([98ba0b7](https://github.com/es-ude/elastic-ai.creator/commit/98ba0b78090c120070e38bf9b0502b3027e0fa33)) - Julian Hoever
- **(qat)** remove noise comments - ([ccc3979](https://github.com/es-ude/elastic-ai.creator/commit/ccc397911d899bdc49b917d0438336e17e37d100)) - Julian Hoever
- **(templates)** reading text from resources returns list[str] - ([361c5a5](https://github.com/es-ude/elastic-ai.creator/commit/361c5a571432f24b8b2be0327fdc1b4edea1c6fe)) - Lukas Einhaus
- **(templates)** use streams for reading templates - ([2af5d2a](https://github.com/es-ude/elastic-ai.creator/commit/2af5d2a41d72406a4abcbb2c55f3c0c01150cad4)) - Lukas Einhaus
- **(templates)** reading text from resources returns list[str] - ([d10178f](https://github.com/es-ude/elastic-ai.creator/commit/d10178f89f4d5e1b24b8860846bd17b72af93ec0)) - Lukas Einhaus
- **(templates)** use streams for reading templates - ([13422aa](https://github.com/es-ude/elastic-ai.creator/commit/13422aaae48eeb22fae01001ee31e3a71d85c337)) - Lukas Einhaus
- **(tests)** move all unit and integration tests in a tests folder - ([8afb751](https://github.com/es-ude/elastic-ai.creator/commit/8afb751a5dc9f7fd4e2fa4a1dd1167682efe590f)) - Julian Hoever
- **(tests)** move last tests out of old test folder structure - ([a3e12c1](https://github.com/es-ude/elastic-ai.creator/commit/a3e12c11df45b4de8babb2f1862ea92a1778c92a)) - Lukas Einhaus
- **(typing)** add missing type annotations - ([c83a746](https://github.com/es-ude/elastic-ai.creator/commit/c83a7466cecfc043dd95800c69b6ee5df8b5bd4f)) - Julian Hoever
- **(vhdl)** using code function to generate code instead of call - ([843ad64](https://github.com/es-ude/elastic-ai.creator/commit/843ad64d33e2018da9c88fd487ccd46fd598c58f)) - Julian Hoever
- **(vhdl)** add missing type annotations and remove unused parts - ([6fb622b](https://github.com/es-ude/elastic-ai.creator/commit/6fb622b3cff4caf8d849c8df8696275fa38fa9bb)) - Julian Hoever
- **(vhdl)** rename call function to code in the language module - ([4cf795e](https://github.com/es-ude/elastic-ai.creator/commit/4cf795ee1cc679ea6b4b7cf51198cb536a5d9af5)) - Julian Hoever
- **(vhdl)** rename call function to code in the precomputed scalar functions and test benches - ([a40553e](https://github.com/es-ude/elastic-ai.creator/commit/a40553e05f64d9bd57473fed4d40b269858ef65f)) - Julian Hoever
- **(vhdl)** fix some mypy errors - ([cd9899f](https://github.com/es-ude/elastic-ai.creator/commit/cd9899f55476cd91993f7276cdda02fc7e3d7b26)) - Julian Hoever
- **(vhdl)** remove CodeModule/CodeComponent - ([89e27dd](https://github.com/es-ude/elastic-ai.creator/commit/89e27dd4e1bd77502d69054c15a3277f0a4b0826)) - Lukas Einhaus
- **(vhdl)** refactor FPHardSigmoidFile - ([3c54418](https://github.com/es-ude/elastic-ai.creator/commit/3c544184877d3416445f064d33bee3e95d78ac31)) - Lukas Einhaus
- **(vhdl)** use VHDLFile for root module generation - ([e2b8423](https://github.com/es-ude/elastic-ai.creator/commit/e2b842310f237ff8802bd92ac4f7f537d9ede707)) - Lukas Einhaus
- **(vhdl)** move files - ([01999c4](https://github.com/es-ude/elastic-ai.creator/commit/01999c4049f21e357390c2fc88a09bfc987d0cb6)) - Lukas Einhaus
- **(vhdl)** rename BaseHWBlockInterface to BaseHWBlock - ([4c69682](https://github.com/es-ude/elastic-ai.creator/commit/4c696826e559603ac54681eae7ff50e34d22a1ac)) - Lukas Einhaus
- **(vhdl)** move classes out of hw_equivalent_layers.__init__ - ([0713211](https://github.com/es-ude/elastic-ai.creator/commit/071321164dee3e9a9db3d113848c6ce3dd960b1c)) - Lukas Einhaus
- **(vhdl)** remove CodeModule/CodeComponent - ([f014414](https://github.com/es-ude/elastic-ai.creator/commit/f014414d9e513303b8f128b4cf87550a04a863d7)) - Lukas Einhaus
- **(vhdl)** refactor FPHardSigmoidFile - ([461ff71](https://github.com/es-ude/elastic-ai.creator/commit/461ff71ad50f22c92dcf430a7b98b708a32a86a8)) - Lukas Einhaus
- **(vhdl)** use VHDLFile for root module generation - ([e386141](https://github.com/es-ude/elastic-ai.creator/commit/e386141d3a318782f5b7793a1672c388d74b5563)) - Lukas Einhaus
- **(vhdl)** move files - ([1006de2](https://github.com/es-ude/elastic-ai.creator/commit/1006de2c4b126e467b91b05236cebdcd40be48df)) - Lukas Einhaus
- **(vhdl)** sort imports - ([2c114f1](https://github.com/es-ude/elastic-ai.creator/commit/2c114f1b8c839ef939051f5a1c5b6d40585908cb)) - Lukas Einhaus
- use new vhdl file class for hard_sigmoid - ([a9e1f6c](https://github.com/es-ude/elastic-ai.creator/commit/a9e1f6ccba2f978b8290be94c754872432d3c311)) - Lukas Einhaus
- use new vhdl file class for hard_sigmoid - ([36cae74](https://github.com/es-ude/elastic-ai.creator/commit/36cae74c068219e96bc3d3eeabbfc87c7ed2e9e8)) - Lukas Einhaus
- move files, simplify, correct types, merge - ([57d3754](https://github.com/es-ude/elastic-ai.creator/commit/57d37541fed53c29ad9e6a665aa72b99fe5a2df0)) - Lukas Einhaus

### Style

- **(vhdl)** introduce template code file interface - ([69fb2b6](https://github.com/es-ude/elastic-ai.creator/commit/69fb2b681497289d230f8d203f5f430c91a3ff54)) - Lukas Einhaus
- **(vhdl)** introduce template code file interface - ([4b233c3](https://github.com/es-ude/elastic-ai.creator/commit/4b233c37c41f231e94a9b6cb800146eb5d0ecb62)) - Lukas Einhaus
- remove deprecated code and move/rename - ([f6b8020](https://github.com/es-ude/elastic-ai.creator/commit/f6b8020f8a5dfc5d9226578efa5f2512b84223e5)) - Lukas Einhaus
- remove deprecated code and move/rename - ([ea9125a](https://github.com/es-ude/elastic-ai.creator/commit/ea9125a649611f260e2c7600fd8619d74f3c5fba)) - Lukas Einhaus
- beautify 5fcfc23c342983a98efc1d527648ef17644c472c - ([a228cc0](https://github.com/es-ude/elastic-ai.creator/commit/a228cc00bf0a44327a858835e9a73531af56e59e)) - github-actions

### Tests

- **(qat)** start splitting large LayersTest TestCase class into smaller classes for each layer - ([905f165](https://github.com/es-ude/elastic-ai.creator/commit/905f165fabf01e0a5897ffc41bff01acac3175b2)) - Julian Hoever
- **(vhdl)** start making network.vhd tests more specific - ([ea260f0](https://github.com/es-ude/elastic-ai.creator/commit/ea260f0bc989a0bc477b425f14364bdb7756a37b)) - Lukas Einhaus
- **(vhdl)** add multiline begin-end extraction for code - ([e958557](https://github.com/es-ude/elastic-ai.creator/commit/e958557fccaabf46d4020cf23fae14e20c4452ee)) - Lukas Einhaus
- **(vhdl)** introduce more fine grained testing for generated vhd files - ([5a27cdc](https://github.com/es-ude/elastic-ai.creator/commit/5a27cdc16ed7b3cbb880c022ac65e1adcfdca563)) - Lukas Einhaus
- **(vhdl)** start making network.vhd tests more specific - ([06506e2](https://github.com/es-ude/elastic-ai.creator/commit/06506e2883a4cb8d7e182a5a7ddbf1a9f6814e1d)) - Lukas Einhaus
- **(vhdl)** add multiline begin-end extraction for code - ([efeeee6](https://github.com/es-ude/elastic-ai.creator/commit/efeeee679973e03e373b4e72f1ef2e411fdbff28)) - Lukas Einhaus
- **(vhdl)** introduce more fine grained testing for generated vhd files - ([c11059c](https://github.com/es-ude/elastic-ai.creator/commit/c11059cd36074ce23285b3f22cafdb412a6077f7)) - Lukas Einhaus

### Ci

- **(gh-workflow)** don't install onnx - ([7b164cd](https://github.com/es-ude/elastic-ai.creator/commit/7b164cdd95a5af672f78c7c22e267c3364fe4d0a)) - Lukas Einhaus
- **(templates)** add commitlint constraints - ([d345351](https://github.com/es-ude/elastic-ai.creator/commit/d345351c96c0ce6d0a5bcf52ae9bca8eacdafd6b)) - Lukas Einhaus
- **(templates)** add commitlint constraints - ([0831b28](https://github.com/es-ude/elastic-ai.creator/commit/0831b281452be446ac8abceba3825ce7a155078b)) - Lukas Einhaus
- adjust commitlint config - ([e518975](https://github.com/es-ude/elastic-ai.creator/commit/e518975728140ab29692bea341ea015cfcfb59df)) - Lukas Einhaus
- temporarily disable commitlint constraints - ([87eaa63](https://github.com/es-ude/elastic-ai.creator/commit/87eaa632c644d70c5ac693b2f5f6aac6a3625acc)) - Lukas Einhaus
- temporarily further relax commitlint - ([08eab5b](https://github.com/es-ude/elastic-ai.creator/commit/08eab5b817d772457dfa983846adb36a8f1b64d3)) - Lukas Einhaus
- clean up pyproject.toml - ([547d724](https://github.com/es-ude/elastic-ai.creator/commit/547d724db5c14392b148c8cf9e0a5714b1052a4d)) - Lukas Einhaus

---
## [0.26.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.26.0..v0.26.1) - 2022-11-30

### Bug Fixes

- **(vhdl)** remove layer_name parameter - ([7a83b1e](https://github.com/es-ude/elastic-ai.creator/commit/7a83b1eed3095a8b7f90438c78ba24bba6e44958)) - Julian Hoever
- **(vhdl)** remove layer_name parameter - ([1bb40cd](https://github.com/es-ude/elastic-ai.creator/commit/1bb40cd0e44f7f207f60ffbb33e8c59f00b64e82)) - Julian Hoever

### Features

- **(vhdl)** implement quantized forward function of the fixed point lstm cell - ([7818e15](https://github.com/es-ude/elastic-ai.creator/commit/7818e15bc6c41454090b77fe5df7a8e7930ab570)) - Julian Hoever
- **(vhdl)** start implementing lstm base module - ([b154ca5](https://github.com/es-ude/elastic-ai.creator/commit/b154ca5525c00f735150c21f64324da87328ba5e)) - Julian Hoever

### Refactoring

- **(examples)** remove examples that are not relevant anymore - ([3b241e2](https://github.com/es-ude/elastic-ai.creator/commit/3b241e2ddfe14a248e411f0b8da9ec6cf85cc8bc)) - Julian Hoever
- **(vhdl)** rename lstm module to lstm_cell - ([97bf791](https://github.com/es-ude/elastic-ai.creator/commit/97bf791a8a3fbab68179f5a9a20e9410c3bcccf7)) - Julian Hoever
- **(vhdl)** remove vhdl formatter that is not used anymore - ([007a8c4](https://github.com/es-ude/elastic-ai.creator/commit/007a8c4ec4c42382390b0af034a2f5f3226fea86)) - Julian Hoever

---
## [0.26.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.25.0..v0.26.0) - 2022-11-23

### Bug Fixes

- **(vhdl)** fix error during integrating to a MLP model - ([0e2b89c](https://github.com/es-ude/elastic-ai.creator/commit/0e2b89c898497f35a2ad840bd3065429799bdf61)) - Chao Qian

### Features

- **(vhdl)** merge from main - ([fefd3ba](https://github.com/es-ude/elastic-ai.creator/commit/fefd3ba4ab1fa8ae9d09bfc6185f906175f7a6ff)) - Chao Qian
- **(vhdl)** make linear layers better timing - ([1c6a3ae](https://github.com/es-ude/elastic-ai.creator/commit/1c6a3aeeeaee929affbb092eb485c1cf7a323355)) - Chao Qian
- **(vhdl)** clean the code - ([d737d02](https://github.com/es-ude/elastic-ai.creator/commit/d737d02122207bcd24f4b7c960b71db095d34a26)) - Chao Qian

### Miscellaneous Chores

- **(gh-workflow)** remove minor python versions from gh workflow - ([fc517a6](https://github.com/es-ude/elastic-ai.creator/commit/fc517a6bb81fb037f2b9d3466d32506aa0573020)) - Julian Hoever
- **(gh-workflow)** remove minor python versions from gh workflow - ([26b8035](https://github.com/es-ude/elastic-ai.creator/commit/26b803589da4d8f60cc7c52dc2a27b97cca88ab9)) - Julian Hoever

---
## [0.25.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.24.0..v0.25.0) - 2022-11-22

### Bug Fixes

- **(vhdl)** remove the layer name in the example file - ([767b5f9](https://github.com/es-ude/elastic-ai.creator/commit/767b5f9c62d493d35e5a294b1363c861d5438fa5)) - Chao Qian
- **(vhdl)** fix small error in the template file - ([fe94518](https://github.com/es-ude/elastic-ai.creator/commit/fe94518ff2e5e44f7c1ff8f9bf8b4ff8f0b5cf41)) - Chao Qian
- **(vhdl)** fix the error from merging braches - ([c386766](https://github.com/es-ude/elastic-ai.creator/commit/c386766ea654852c5ad5254cefc1fab28f544c66)) - Chao Qian

### Features

- **(vhdl)** add expand_template function that fills string templates instead of format strings - ([eb9ee98](https://github.com/es-ude/elastic-ai.creator/commit/eb9ee987f73ffb26e8280ec3c32b32e38896d3c1)) - Julian Hoever
- **(vhdl)** apply the expand_template function to the already existing templates - ([c958f54](https://github.com/es-ude/elastic-ai.creator/commit/c958f545f4c2cf2414a007753b416ec73c410458)) - Julian Hoever

---
## [0.24.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.23.0..v0.24.0) - 2022-11-22

### Bug Fixes

- **(vhdl)** fix wrong return type - ([eb53ed9](https://github.com/es-ude/elastic-ai.creator/commit/eb53ed972ec9078f6c405ecd7c92043eaf8ed419)) - Julian Hoever
- **(vhdl)** remove duplicated key - ([5a4bcd6](https://github.com/es-ude/elastic-ai.creator/commit/5a4bcd6fb6de9cff6c639866db1dd50918f3039b)) - Julian Hoever

### Features

- **(vhdl)** implement FixedPointHardTanh layer - ([ed72810](https://github.com/es-ude/elastic-ai.creator/commit/ed728101fb596a08e1a76d936d04306a066c50b5)) - Julian Hoever
- **(vhdl)** start implementing lstm base layer - ([39ce891](https://github.com/es-ude/elastic-ai.creator/commit/39ce891d56be59d5a20a36889b0e9c2f13e00bd1)) - Julian Hoever
- **(vhdl)** implement and test lstm cell base class and start implementing fp lstm cell - ([f458fb6](https://github.com/es-ude/elastic-ai.creator/commit/f458fb6c216385a119774a3f98788941e13ed5c9)) - Julian Hoever
- **(vhdl)** add layer_id parameter to build function and set it to a unique value during translation - ([cfdf949](https://github.com/es-ude/elastic-ai.creator/commit/cfdf9492190e24230293e3b0b1b312bfc9710952)) - Julian Hoever

### Miscellaneous Chores

- **(gh-workflow)** update action versions and remove usage of set-output - ([d106116](https://github.com/es-ude/elastic-ai.creator/commit/d1061167f9da09f7aa2a191340652ae56e3335e0)) - Julian Hoever
- **(gh-workflow)** set correct commit action - ([a7a2439](https://github.com/es-ude/elastic-ai.creator/commit/a7a2439d8b8a1358983d18c22de6e57820757d82)) - Julian Hoever
- **(gh-workflow)** set correct parameters for the commit action - ([373ffd2](https://github.com/es-ude/elastic-ai.creator/commit/373ffd20d331152cedde6083d72fdb40d83c741d)) - Julian Hoever
- **(gh-workflow)** set correct parameters for the commit action - ([0193074](https://github.com/es-ude/elastic-ai.creator/commit/019307473d639c14f3949b5e7d42be4cc14f655f)) - Julian Hoever

### Refactoring

- **(vhdl)** move common helper functions to a separate utils.py module - ([b459f0a](https://github.com/es-ude/elastic-ai.creator/commit/b459f0af4b8b9884c03277928d7a0437b68f9716)) - Julian Hoever
- **(vhdl)** move OperationType type in the typing module - ([bf0d3fe](https://github.com/es-ude/elastic-ai.creator/commit/bf0d3feacdfdf461527056b8a24aec63907a2578)) - Julian Hoever

---
## [0.23.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.22.0..v0.23.0) - 2022-11-15

### Documentation

- **(vhdl)** change documentation - ([d3fb540](https://github.com/es-ude/elastic-ai.creator/commit/d3fb5402c7acb09cee3df535671f22d5011f2f47)) - Chao Qian

### Features

- **(vhdl)** merge main to current working branch - ([35db3c5](https://github.com/es-ude/elastic-ai.creator/commit/35db3c56608493c6b33d05e0c2250cedb0374c8e)) - Chao Qian
- **(vhdl)** enable multiple linear layers in the same model, by adding layer_name - ([3a99a30](https://github.com/es-ude/elastic-ai.creator/commit/3a99a3059dd53b913e7d619cbce28014007bf854)) - Chao Qian
- **(vhdl)** remove the previous linear_1d implementation - ([0f1b9aa](https://github.com/es-ude/elastic-ai.creator/commit/0f1b9aa2f1c12f5c0fc1fe6a3db482f40041c057)) - Chao Qian

---
## [0.22.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.21.0..v0.22.0) - 2022-11-13

### Documentation

- **(vhdl)** add missing parameter in docstring of the translate_model function - ([458a02c](https://github.com/es-ude/elastic-ai.creator/commit/458a02c38402a0860500d5821b68890fcc78c01a)) - Julian Hoever

### Features

- **(vhdl)** raise an exception if the build folder already exists - ([d09bfa1](https://github.com/es-ude/elastic-ai.creator/commit/d09bfa105d909b58432cf8883ee55a6b11639add)) - Julian Hoever

---
## [0.21.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.20.1..v0.21.0) - 2022-11-13

### Features

- **(vhdl)** add fp_linear_component and its template unittest is passed - ([6e97316](https://github.com/es-ude/elastic-ai.creator/commit/6e973168ca244e4cf407c48b31406d2eed73b4b0)) - Chao Qian
- **(vhdl)** add fp_linear_module and test passed - ([241fd65](https://github.com/es-ude/elastic-ai.creator/commit/241fd652495d6ce582873f1bcc297302f3d61764)) - Chao Qian
- **(vhdl)** add fp_linear build function, and test passed - ([ffcbb1d](https://github.com/es-ude/elastic-ai.creator/commit/ffcbb1d57408ad03e91bd1228bc6d3289f1d0c66)) - Chao Qian
- **(vhdl)** add default build function mapping and small changes - ([b1d6f2a](https://github.com/es-ude/elastic-ai.creator/commit/b1d6f2ac1040e63781d5f4af7ee29e486d9b6d69)) - Chao Qian
- **(vhdl)** check the component interface - ([53791c5](https://github.com/es-ude/elastic-ai.creator/commit/53791c5eb9a72793b16a0a41eb79ed8932b8e32d)) - Chao Qian
- **(vhdl)** add fixed point relu to translator - ([80935ce](https://github.com/es-ude/elastic-ai.creator/commit/80935ce550a2e99267a55b41ad272906faf211a5)) - Chao Qian
- **(vhdl)** add default build mapping for fp_hard_sigmoid and fp_relu - ([c9c4d9f](https://github.com/es-ude/elastic-ai.creator/commit/c9c4d9f329ed2c56d47f2b698dbe1d3b34c1c8a5)) - Chao Qian

### Refactoring

- **(vhdl)** change the interface name of the template - ([a693041](https://github.com/es-ude/elastic-ai.creator/commit/a693041a050ef77828a4f4dba791b0a38a845184)) - Chao Qian
- **(vhdl)** get rid of some comments - ([ced1b12](https://github.com/es-ude/elastic-ai.creator/commit/ced1b127031c02d8576ccc35fdd9f143017c3368)) - Chao Qian

### Tests

- **(vhdl)** add test coverage of relu component - ([88e0d10](https://github.com/es-ude/elastic-ai.creator/commit/88e0d10d4cb0a64ac397cee1a9e42db9184c6139)) - Chao Qian

---
## [0.20.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.20.0..v0.20.1) - 2022-11-10

### Bug Fixes

- **(vhdl)** fix incompatible signature of the forward function - ([ff6c165](https://github.com/es-ude/elastic-ai.creator/commit/ff6c165cd0bf17477051548018b791809fff33c9)) - Julian Hoever

### Refactoring

- **(vhdl)** small change of the FixedPointFactory type - ([c7629bd](https://github.com/es-ude/elastic-ai.creator/commit/c7629bd05764de09f03fd8445437dee671518d38)) - Julian Hoever
- **(vhdl)** remove usage of deprecated assertEquals function - ([6a6f4f3](https://github.com/es-ude/elastic-ai.creator/commit/6a6f4f3af28735e27fc70b51f857324cd1ead7ef)) - Julian Hoever

---
## [0.20.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.19.0..v0.20.0) - 2022-11-08

### Documentation

- **(vhdl)** add documentation for the quantized_modules package - ([9da4a0d](https://github.com/es-ude/elastic-ai.creator/commit/9da4a0d380304a7ab8834049ad93bed547816ddb)) - Julian Hoever

### Features

- **(examples)** add example using quantized modules to verify the current state of the translator - ([0c55e00](https://github.com/es-ude/elastic-ai.creator/commit/0c55e00657c0d260766155995b75f25bff642e24)) - Julian Hoever
- **(vhdl)** integrate fixed point hard sigmoid to the translator - ([0a07cee](https://github.com/es-ude/elastic-ai.creator/commit/0a07ceeb3d238456dad08448b543f4a075873322)) - Julian Hoever

### Refactoring

- **(examples)** rename example to fit its actual content - ([9ac5a83](https://github.com/es-ude/elastic-ai.creator/commit/9ac5a83a558887d0cf4830a6f7ba94ede92de594)) - Julian Hoever
- **(examples)** remove unused import - ([fbb684d](https://github.com/es-ude/elastic-ai.creator/commit/fbb684daaeb8376ec7a56b413959cb9e9f2dc600)) - Julian Hoever
- **(vhdl)** change name of the output_quant parameter to output_dequant to be more precise - ([3186b5b](https://github.com/es-ude/elastic-ai.creator/commit/3186b5b848e4b7be8e0bc0d94a1897722d2e2397)) - Julian Hoever

### Tests

- **(vhdl)** add tests to test the quant and dequant parameters of the LinearBase class - ([ad49bc6](https://github.com/es-ude/elastic-ai.creator/commit/ad49bc68ef5f38e6047d569d8e90513f50698a27)) - Julian Hoever

---
## [0.19.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.18.0..v0.19.0) - 2022-11-05

### Features

- **(vhdl)** merge translate_model and generate_code functions - ([c12562e](https://github.com/es-ude/elastic-ai.creator/commit/c12562ee4a55c61b5ef82b5ef37568fe32e8f525)) - Julian Hoever

---
## [0.18.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.17.0..v0.18.0) - 2022-11-04

### Features

- **(examples)** add simulated fixed point inference to the example - ([4f81d8d](https://github.com/es-ude/elastic-ai.creator/commit/4f81d8d3d44f1c677fc1a12edf94b7b614d72efb)) - Julian Hoever
- **(vhdl)** refactoring and start implementing hard sigmoid activation function - ([ff94c9d](https://github.com/es-ude/elastic-ai.creator/commit/ff94c9dd1d1297f02e82a0d1f7f203f80c8d2732)) - Julian Hoever
- **(vhdl)** fix wrong calculation of fixed point values and add quantized forward functions - ([93046d3](https://github.com/es-ude/elastic-ai.creator/commit/93046d3b93d1a977c4106cf56e7f98847a47aa00)) - Julian Hoever
- **(vhdl)** implement a version of relu for qat and quantized inference - ([ddd9607](https://github.com/es-ude/elastic-ai.creator/commit/ddd9607e8dbf333817112dfe24f795ac717f609e)) - Julian Hoever
- **(vhdl)** use fixed point hard sigmoid and relu in the example - ([90350b9](https://github.com/es-ude/elastic-ai.creator/commit/90350b91b9ac917c8c1f0ab50c2744fb09671947)) - Julian Hoever
- **(vhdl)** implement evaluator for simulation of a quantized inference - ([353e82e](https://github.com/es-ude/elastic-ai.creator/commit/353e82e798359c3b15a42a02dcdc63e071b2d34e)) - Julian Hoever
- **(vhdl)** implement evaluator that evaluates a model according to a given metric - ([a0b089a](https://github.com/es-ude/elastic-ai.creator/commit/a0b089ad1f7c32acc0c4522bf830080442e8414d)) - Julian Hoever
- **(vhdl)** add clamp to min or max fixed point integer for overflowing values - ([ca3fc19](https://github.com/es-ude/elastic-ai.creator/commit/ca3fc19aec062d4de34a4698c9e0a9351b41c761)) - Julian Hoever

### Refactoring

- **(examples)** rename fixed point linear layer example - ([59216da](https://github.com/es-ude/elastic-ai.creator/commit/59216da6973daca87e80c105513586df1c682ba6)) - Julian Hoever
- **(vhdl)** create a better module structure - ([b2dfeee](https://github.com/es-ude/elastic-ai.creator/commit/b2dfeee795d980aabc2822e3b1470f2e41d63416)) - Julian Hoever
- **(vhdl)** removed unfinished fixed point configuration finder - ([fa6dc44](https://github.com/es-ude/elastic-ai.creator/commit/fa6dc44e0a02f57993f08b381b62297c2682b167)) - Julian Hoever
- **(vhdl)** small changes to make the code easier to understand - ([7168ba1](https://github.com/es-ude/elastic-ai.creator/commit/7168ba145243616d247006f56d99de3c21e91401)) - Julian Hoever
- **(vhdl)** make floating point values more explicit - ([231b903](https://github.com/es-ude/elastic-ai.creator/commit/231b903127dcdc0b90bc6a4e29ccd29543033935)) - Julian Hoever
- **(vhdl)** remove unused line of code - ([f020554](https://github.com/es-ude/elastic-ai.creator/commit/f020554dc3f1ae6ed4ac025711e6ec1025ba8964)) - Julian Hoever

### Tests

- **(vhdl)** write tests for the evaluators and do some refactoring - ([2641578](https://github.com/es-ude/elastic-ai.creator/commit/2641578eb1820793e4e2117563dc11607707e11d)) - Julian Hoever

---
## [0.17.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.16.0..v0.17.0) - 2022-10-22

### Features

- **(examples)** visualize model parameters - ([5e1b4fc](https://github.com/es-ude/elastic-ai.creator/commit/5e1b4fc4c827c55d19cb9bc4206f706bcc737fba)) - Julian Hoever

---
## [0.16.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.15.0..v0.16.0) - 2022-10-22

### Bug Fixes

- **(vhdl)** fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase - ([da11356](https://github.com/es-ude/elastic-ai.creator/commit/da113561d69158ccc2a9266adb1eddcc79b1cb7d)) - Julian Hoever

### Features

- **(example)** add the ability to plot the model parameters - ([b1b0b5e](https://github.com/es-ude/elastic-ai.creator/commit/b1b0b5e7697992c4c53825c739e2fb2dcc903dac)) - Julian Hoever
- **(examples)** start implementing an example for learning a simple logic function - ([6cff6de](https://github.com/es-ude/elastic-ai.creator/commit/6cff6deccd5c2080e930d93f5e145e4d7ea6a41e)) - Julian Hoever
- **(examples)** commit current state of the fixed point linear example - ([9b8ecae](https://github.com/es-ude/elastic-ai.creator/commit/9b8ecae971bc1dedabf17e79272008a3cbfb5123)) - Julian Hoever
- **(vhdl)** add function to get attribute names of an object matching a regex - ([acc8e29](https://github.com/es-ude/elastic-ai.creator/commit/acc8e29e2771d5642e1371af6fb3c44f83b5ebc7)) - Julian Hoever
- **(vhdl)** implement custom linear layers that allows to do fixed point calculations - ([c2364f6](https://github.com/es-ude/elastic-ai.creator/commit/c2364f6182bb8406e90a78d632bc868537705fd2)) - Julian Hoever
- **(vhdl)** add a type for fixed point factories - ([53b8499](https://github.com/es-ude/elastic-ai.creator/commit/53b84991671832c2e7fa24e61d927b7c039832d9)) - Julian Hoever
- **(vhdl)** make base linear package private - ([b3cfa55](https://github.com/es-ude/elastic-ai.creator/commit/b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554)) - Julian Hoever
- **(vhdl)** move tracing example to example folder - ([b942155](https://github.com/es-ude/elastic-ai.creator/commit/b942155a240a6f34f0f02361b6631b431a448443)) - Julian Hoever
- **(vhdl)** add feature to automatically derive fixed point parameters from a factory - ([70618d5](https://github.com/es-ude/elastic-ai.creator/commit/70618d512718efd7e718491af52e1acbc6c86622)) - Julian Hoever
- **(vhdl)** move the input, weight and output quantization to the linear layer - ([0c8b259](https://github.com/es-ude/elastic-ai.creator/commit/0c8b259ef688c606ebd4f8486ef7b6f48e0f8713)) - Julian Hoever
- **(vhdl)** implement qat for linear layer - ([d3ba49e](https://github.com/es-ude/elastic-ai.creator/commit/d3ba49e266b2931c1b16677dd91f17a75f091501)) - Julian Hoever

### Miscellaneous Chores

- **(vhdl)** skip some tests that are not relevant at the moment - ([2dbfd55](https://github.com/es-ude/elastic-ai.creator/commit/2dbfd55a805166e97b640fafd5ebc1214288a863)) - Julian Hoever

### Refactoring

- **(creator)** remove unused typevars and change typing - ([2770991](https://github.com/es-ude/elastic-ai.creator/commit/2770991c00a9395697180fcec98e733164efde24)) - Julian Hoever

---
## [0.15.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.14.0..v0.15.0) - 2022-09-29

### Features

- **(vhdl)** implement clipped fixed point representation - ([8e53506](https://github.com/es-ude/elastic-ai.creator/commit/8e53506fce0ba5adaa124ccd61de3b340bf1c95f)) - Julian Hoever

---
## [0.14.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.13.0..v0.14.0) - 2022-09-28

### Bug Fixes

- **(vhdl)** reimplement unsigned_int_values_to_fixed_point function - ([cdd069e](https://github.com/es-ude/elastic-ai.creator/commit/cdd069e6adffa882bb34fea2b7179891c282045b)) - Julian Hoever

### Features

- **(vhdl)** implement automatic derivation of fixed point parameters in the lstm example - ([504008d](https://github.com/es-ude/elastic-ai.creator/commit/504008d7ef3f402f8476bb77f02a4a37176d229e)) - Julian Hoever
- **(vhdl)** implement signed fixed point integer to FixedPoint object - ([0a2fc79](https://github.com/es-ude/elastic-ai.creator/commit/0a2fc7952dc13ea48c749856bf809a5540166598)) - Julian Hoever
- **(vhdl)** start implementing fixed point evaluator - ([0f9c62a](https://github.com/es-ude/elastic-ai.creator/commit/0f9c62a38f9df6ee4e84f1a3b5524df03511b438)) - Julian Hoever
- **(vhdl)** working further on the fixed point configuration finder - ([beb9da0](https://github.com/es-ude/elastic-ai.creator/commit/beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d)) - Julian Hoever
- **(vhdl)** implement from_unsigned_int and from_signed_int function and remove unused function - ([aca77f5](https://github.com/es-ude/elastic-ai.creator/commit/aca77f5eac396f21821b07706ff250b2589dd037)) - Julian Hoever

---
## [0.13.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.12.1..v0.13.0) - 2022-09-10

### Features

- **(gh-workflow)** explicitly set poetry version - ([82d202f](https://github.com/es-ude/elastic-ai.creator/commit/82d202f0229e7931fc7371f69abe0d1fe3a58134)) - Julian Hoever
- **(vhdl)** remove translatable protocol - ([37412e8](https://github.com/es-ude/elastic-ai.creator/commit/37412e87d89d16c9159cf12ef00032343119100c)) - Julian Hoever
- **(vhdl)** remove translatable protocol - ([ca52a92](https://github.com/es-ude/elastic-ai.creator/commit/ca52a92d1bdc017773b872eaa5011b5117394472)) - Julian Hoever

### Refactoring

- **(vhdl)** remove translatable protocol - ([ef5f8fd](https://github.com/es-ude/elastic-ai.creator/commit/ef5f8fd4b914cbaf9ff4369aaa72437a1b68f5d3)) - Lukas Einhaus

### Style

- beautify 2024c7e9a8aa9ed2487f58586aa41beabd6f63d2 - ([7a58041](https://github.com/es-ude/elastic-ai.creator/commit/7a58041b71ccd258d9fcb16b1ac1a15be32e212d)) - github-actions

---
## [0.12.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.12.0..v0.12.1) - 2022-08-29

### Bug Fixes

- **(qat)** reimplement binarize - ([9bbccdd](https://github.com/es-ude/elastic-ai.creator/commit/9bbccddfc6ce6c2b928166cdfaf1112b294dba17)) - Lukas Einhaus

### Style

- beautify 7a60a043e83aedcdf281ec9357ee9f274aca59dd - ([0723cb1](https://github.com/es-ude/elastic-ai.creator/commit/0723cb12e8fc6290403efd68b6d552a81ad69a99)) - github-actions

---
## [0.12.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.11.1..v0.12.0) - 2022-08-26

### Bug Fixes

- **(vhdl)** fix calculation of the addr_width of the linear1d layer - ([6fa2b2a](https://github.com/es-ude/elastic-ai.creator/commit/6fa2b2a3bc83d3a51eb955d1464501662f6676a8)) - Julian Hoever
- **(vhdl)** pre-add input-hidden and hidden-hidden bias - ([750941c](https://github.com/es-ude/elastic-ai.creator/commit/750941c3150cabefa2f393f6b12105a358a70f7f)) - Julian Hoever
- **(vhdl)** add changes from Chao after testing the translator - ([5a5d532](https://github.com/es-ude/elastic-ai.creator/commit/5a5d5325a3f598e0163d4eac0601b5961c2f5780)) - Julian Hoever
- **(vhdl)** fix test - ([c125bf1](https://github.com/es-ude/elastic-ai.creator/commit/c125bf16297ee9e39660ee904ab54268e8901d48)) - Julian Hoever
- **(vhdl)** remove unused work library - ([c68fd9d](https://github.com/es-ude/elastic-ai.creator/commit/c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0)) - Julian Hoever
- **(vhdl)** remove some comments - ([13cc1a1](https://github.com/es-ude/elastic-ai.creator/commit/13cc1a1ade14ccc7aa686523270dec20936ed14d)) - Julian Hoever

### Documentation

- **(readme)** update documentation according the newly added linear1d layer - ([41e2486](https://github.com/es-ude/elastic-ai.creator/commit/41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b)) - Julian Hoever
- **(readme)** small changes of the documentation - ([9e7699c](https://github.com/es-ude/elastic-ai.creator/commit/9e7699ce617581f67f85cf4ef7d945d99df241be)) - Julian Hoever
- **(readme)** move translator documentation to the vhdl package - ([9a90949](https://github.com/es-ude/elastic-ai.creator/commit/9a90949528978ff4732f585986a71cedd44e82a5)) - Julian Hoever
- **(vhdl)** adapt diagrams to the latest changes - ([c1750eb](https://github.com/es-ude/elastic-ai.creator/commit/c1750eb19f92a705f8f36ccefc9729d3545f0743)) - Julian Hoever

### Features

- **(vhdl)** insert values in the updated lstm.vhd template - ([4d9dccb](https://github.com/es-ude/elastic-ai.creator/commit/4d9dccbdb11afebb466f476c22539828bf5458b1)) - Julian Hoever
- **(vhdl)** make work library name customizable - ([95fd8aa](https://github.com/es-ude/elastic-ai.creator/commit/95fd8aa0d7e512aeb04893de2df2e58cc4b3e641)) - Julian Hoever

### Miscellaneous Chores

- **(docker)** add a dockerfile - ([e2f54b3](https://github.com/es-ude/elastic-ai.creator/commit/e2f54b373bf26ee6c94b0c0c448b6e34affb2e64)) - Lukas Einhaus

### Refactoring

- **(vhdl)** simplify code and reuse components - ([37cffd7](https://github.com/es-ude/elastic-ai.creator/commit/37cffd76953e1f7756de8ec7ebc5b356fb89f1ad)) - Julian Hoever

---
## [0.11.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.11.0..v0.11.1) - 2022-08-18

### Bug Fixes

- **(qat)** remove deprecated threshold and codomain properties - ([5db9669](https://github.com/es-ude/elastic-ai.creator/commit/5db9669fc3942851e65607a869bb822430df7836)) - Lukas Einhaus

### Documentation

- **(readme)** add documentation on how the translator works - ([91ebea3](https://github.com/es-ude/elastic-ai.creator/commit/91ebea3fb7e7883f56b2cd9152769d151449a49a)) - Julian Hoever

### Features

- **(examples)** add linear layer to the translation example - ([5f1e1db](https://github.com/es-ude/elastic-ai.creator/commit/5f1e1db8da7ce533cb592d56ca97e25ca563a60e)) - Julian Hoever
- **(vhdl)** implement the translation of a linear1d layer - ([b627e78](https://github.com/es-ude/elastic-ai.creator/commit/b627e780d054adcdd89009d87aa33fa31c913504)) - Julian Hoever
- **(vhdl)** add an easy way to get a fixed point factory - ([d98ff03](https://github.com/es-ude/elastic-ai.creator/commit/d98ff0351f739859ed668a2ec295421e29fd24ec)) - Julian Hoever

### Refactoring

- **(vhdl)** change naming of the translator components - ([fdf5586](https://github.com/es-ude/elastic-ai.creator/commit/fdf5586da727542be6bfab57fba4a98d8ec482d7)) - Julian Hoever
- **(vhdl)** change naming for better understanding - ([17d8a3d](https://github.com/es-ude/elastic-ai.creator/commit/17d8a3d89dcbbb4882c62953ddb928d268945852)) - Julian Hoever
- **(vhdl)** small naming changes - ([fd5c9b4](https://github.com/es-ude/elastic-ai.creator/commit/fd5c9b4f9fccb95b9fd4a0223e87a791fd02224c)) - Julian Hoever

### Style

- beautify 52e7e3e55053a9e95e786bf899056148753cddfc - ([a5c17b4](https://github.com/es-ude/elastic-ai.creator/commit/a5c17b428c20f8c55b7c9350e5d9a33ef8b76822)) - github-actions

### Tests

- **(vhdl)** add tests for rom and linear1d component - ([fc1f20e](https://github.com/es-ude/elastic-ai.creator/commit/fc1f20e30bf91f6aa96ee800e2e66aa5b3c217ad)) - Julian Hoever

### Build

- **(gh-workflow)** perform tests with more verbose output - ([8d7b50b](https://github.com/es-ude/elastic-ai.creator/commit/8d7b50b7ae2c6d513f67027e5f209cf3115e0964)) - Julian Hoever

---
## [0.11.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.10.1..v0.11.0) - 2022-08-11

### Bug Fixes

- **(examples)** use LSTMTranslationArguments object instead of a dictionary - ([98a4d97](https://github.com/es-ude/elastic-ai.creator/commit/98a4d97f8fbd217f67ed4009ab63ccc4705f720d)) - Julian Hoever
- **(vhdl)** fix test - ([528910c](https://github.com/es-ude/elastic-ai.creator/commit/528910cf3fe28958ebb7b246104e83df77bbf3f4)) - Julian Hoever
- **(vhdl)** fix wrong pytorch lstm cell class path - ([85a733c](https://github.com/es-ude/elastic-ai.creator/commit/85a733cb5ff821bb602b5021f6438b7d5909382e)) - Julian Hoever
- **(vhdl)** fix mypy typing errors - ([e1dba31](https://github.com/es-ude/elastic-ai.creator/commit/e1dba317585c269ad58719184fb4764cc66485ae)) - Julian Hoever
- **(vhdl)** rename LSTMCell translatable to LSTM - ([e05cd04](https://github.com/es-ude/elastic-ai.creator/commit/e05cd042daf0420b2046607e00eeef3606a6defb)) - Julian Hoever
- **(vhdl)** remove print call - ([55164b7](https://github.com/es-ude/elastic-ai.creator/commit/55164b78c61f37f4cdadde0385965ee540e4f555)) - Julian Hoever

### Documentation

- **(readme)** fix commands of install dev dependencies - ([870e2de](https://github.com/es-ude/elastic-ai.creator/commit/870e2de30f48223d8005bcf1240b624ebb314ad7)) - Tianheng Ling
- **(vhdl)** add some docstrings to the functions of the translator - ([6f9215e](https://github.com/es-ude/elastic-ai.creator/commit/6f9215e5fc35287517d884a702bf887d7a09aa7f)) - Julian Hoever

### Features

- **(examples)** add an example using the vhdl translator for pytorch - ([395adcd](https://github.com/es-ude/elastic-ai.creator/commit/395adcd3e843b7f55f6156ba183dc8800055ef51)) - Julian Hoever
- **(vhdl)** implementation of a LSTMCell class that can be translated to VHDL - ([ace37fe](https://github.com/es-ude/elastic-ai.creator/commit/ace37fe4b215327bc5b43344ffcd0c44a4822dda)) - Julian Hoever
- **(vhdl)** add ability to pass kwargs to the translate function of a translatable layer - ([196812e](https://github.com/es-ude/elastic-ai.creator/commit/196812eecd0dc49a1b8c2d6675b9018ca07e003e)) - Julian Hoever
- **(vhdl)** add a protocol specify a translatable layer - ([0fa966e](https://github.com/es-ude/elastic-ai.creator/commit/0fa966e7f99ef2adb19321b3ca92202616b4c0a2)) - Julian Hoever
- **(vhdl)** introduce translation arguments - ([2c3a8c7](https://github.com/es-ude/elastic-ai.creator/commit/2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f)) - Julian Hoever
- **(vhdl)** abstract LSTM cell takes float weights instead of FixedPoint weights - ([a5818cc](https://github.com/es-ude/elastic-ai.creator/commit/a5818cc0edd918ef3ca49e843738823e988bfd79)) - Julian Hoever
- **(vhdl)** add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell - ([baca5bb](https://github.com/es-ude/elastic-ai.creator/commit/baca5bb6c22692cf9bfc02a9147711b8869930fd)) - Julian Hoever
- **(vhdl)** use __init__ files to simplify the usage - ([3cc07ee](https://github.com/es-ude/elastic-ai.creator/commit/3cc07ee048a349ef5a6a5383dcd829d64b48de2d)) - Julian Hoever
- **(vhdl)** implementation of the mapping of a torch module to the corresponding build function - ([b076fa3](https://github.com/es-ude/elastic-ai.creator/commit/b076fa32cef3c64f8fcc45df24814f4333c90b5c)) - Julian Hoever
- **(vhdl)** first untested draft for the pytorch translator - ([7e59462](https://github.com/es-ude/elastic-ai.creator/commit/7e5946259381af397e1ccd25006815af8256026f)) - Julian Hoever
- **(vhdl)** add the ability to infer the build function from a given layer object or type - ([306df14](https://github.com/es-ude/elastic-ai.creator/commit/306df1427177d15c1b1e2c59b2e774a2a6e2c471)) - Julian Hoever
- **(vhdl)** implement a more functional build function mapping - ([1425e03](https://github.com/es-ude/elastic-ai.creator/commit/1425e0304cf35617106199936d3b014c0d8ca483)) - Julian Hoever
- **(vhdl)** pass an DTO to a translatable instead of raw arguments to fix typing errors - ([4738725](https://github.com/es-ude/elastic-ai.creator/commit/4738725d09ca9114064c4c42dd2818fc6d5c973b)) - Julian Hoever
- **(vhdl)** pass an DTO to a translatable instead of raw arguments to fix typing errors - ([2c33869](https://github.com/es-ude/elastic-ai.creator/commit/2c33869cce5bed725a90ea3a4980bc026aec1ac4)) - Julian Hoever
- **(vhdl)** add LSTMCellTranslationArguments to __init__.py file - ([061ead4](https://github.com/es-ude/elastic-ai.creator/commit/061ead404dc82ddc79ac75c155328ad5733eb04a)) - Julian Hoever
- **(vhdl)** change build function mapping to a different approach - ([b1b79b2](https://github.com/es-ude/elastic-ai.creator/commit/b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e)) - Julian Hoever
- **(vhdl)** make build function mapping more general so that it can be reused for other frameworks - ([3369d7f](https://github.com/es-ude/elastic-ai.creator/commit/3369d7fb6a7d08930514a7c0553c9efe65fc54b9)) - Julian Hoever
- **(vhdl)** removed the possibility to get a build function from a type - ([dbc2e8f](https://github.com/es-ude/elastic-ai.creator/commit/dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2)) - Julian Hoever
- **(vhdl)** change translation from LSTMCell to LSTM - ([5e4f1cf](https://github.com/es-ude/elastic-ai.creator/commit/5e4f1cff380fabd3685660a0c279b9098c4ef278)) - Julian Hoever
- **(vhdl)** adapt the example to the changes of the translation - ([6a5644e](https://github.com/es-ude/elastic-ai.creator/commit/6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e)) - Julian Hoever

### Refactoring

- **(examples)** change the name of an example - ([606c0a3](https://github.com/es-ude/elastic-ai.creator/commit/606c0a30e37e5bd7d7ddc5529c770594debd7605)) - Julian Hoever
- **(typing)** use better typing - ([86a019d](https://github.com/es-ude/elastic-ai.creator/commit/86a019d6d3db8696850b65047481e9566da66cd8)) - Julian Hoever
- **(typing)** vhdl module type is now an iterable instead of an iterator to be more flexible - ([1b471ca](https://github.com/es-ude/elastic-ai.creator/commit/1b471ca3a8f5b3ff3c7c28e105ae3f7f2419367d)) - Julian Hoever
- **(vhdl)** remove custom template mapping, fixed point args and use protocols instead of abc - ([fed2658](https://github.com/es-ude/elastic-ai.creator/commit/fed26585c8ccd123e590476b8e0a8ec4df8891f6)) - Julian Hoever
- **(vhdl)** change typings and Translatable yield VHDLComponent - ([eedacb1](https://github.com/es-ude/elastic-ai.creator/commit/eedacb16afaf805eb6a990aa1ad40273722e02a3)) - Julian Hoever
- **(vhdl)** correct name of a test - ([73d360f](https://github.com/es-ude/elastic-ai.creator/commit/73d360f9f3c9fc6fdf5380ff45c947b49f475199)) - Julian Hoever
- **(vhdl)** change some names to make the code more understandable - ([a8d8b0c](https://github.com/es-ude/elastic-ai.creator/commit/a8d8b0c2fd3a27911a530db20dc3596113fc80e8)) - Julian Hoever
- **(vhdl)** remove empty module - ([03edaca](https://github.com/es-ude/elastic-ai.creator/commit/03edaca097759ff381b012f757631662c4b5fe3a)) - Julian Hoever

### Tests

- **(vhdl)** add tests for the abstract LSTMCell layer - ([643a91f](https://github.com/es-ude/elastic-ai.creator/commit/643a91fa8f569fb002200039a253c9a9a79e5373)) - Julian Hoever
- **(vhdl)** add tests for the translator and the lstm_cell build function - ([a92987d](https://github.com/es-ude/elastic-ai.creator/commit/a92987dfff0e3e2ab7646e984d5309383a0f9681)) - Julian Hoever
- **(vhdl)** add test that should pass in the future to check the correct layer ordering - ([3e5b452](https://github.com/es-ude/elastic-ai.creator/commit/3e5b45266454ce8df5858990314e8702a0db0345)) - Julian Hoever
- **(vhdl)** add tests for the build function mapping - ([4e4fee2](https://github.com/es-ude/elastic-ai.creator/commit/4e4fee2971a4131174dc6286ff85d9f4e0795611)) - Julian Hoever
- **(vhdl)** fixed tests of the build function mapping - ([7885cdb](https://github.com/es-ude/elastic-ai.creator/commit/7885cdb30a413070816442c0e6daf2c0400b2743)) - Julian Hoever

---
## [0.10.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.10.0..v0.10.1) - 2022-06-29

### Bug Fixes

- **(gh-workflow)** try to fix the error with the semantic release tool - ([bc115f8](https://github.com/es-ude/elastic-ai.creator/commit/bc115f899bd85e720448bfa67fe9964bb56c594b)) - Julian Hoever
- **(gh-workflow)** fix error in the main.yml - ([4a6ff5e](https://github.com/es-ude/elastic-ai.creator/commit/4a6ff5e61f35661a3ef83ce4335c109333834d6d)) - Julian Hoever

### Miscellaneous Chores

- **(gh-workflow)** try to fix/investigate the error with the semantic release tool - ([7697877](https://github.com/es-ude/elastic-ai.creator/commit/7697877c44fa382bf0fd3838077078d61b5117dc)) - Julian Hoever

---
## [0.10.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.9.0..v0.10.0) - 2022-06-26

### Documentation

- **(readme)** remove compile instructions for onnx as they are no longer needed - ([3bee70a](https://github.com/es-ude/elastic-ai.creator/commit/3bee70abe4185a0a6708ffc998bdc74004f90b8a)) - Julian Hoever
- **(vhdl)** add a docstring with an example to the FixedPoint class - ([961d766](https://github.com/es-ude/elastic-ai.creator/commit/961d76678d366730f57bbd69b43c38124c003bf7)) - Julian Hoever

### Features

- **(vhdl)** format_vhdl function blocks the process until formatting is complete - ([a8a1bd0](https://github.com/es-ude/elastic-ai.creator/commit/a8a1bd0e7a4db075d0cef4a9eb125a860a697719)) - Julian Hoever

### Miscellaneous Chores

- **(gh-workflow)** removing the compilation steps for onnx as they are no longer needed - ([1118ee4](https://github.com/es-ude/elastic-ai.creator/commit/1118ee4ad89713a56d8a36fb93be46f0a2a33a32)) - Julian Hoever
- **(pyproject)** update numpy, onnx and add pre-commit to dev dependencies - ([a23c00a](https://github.com/es-ude/elastic-ai.creator/commit/a23c00ad3faef0ed5e2318f83553ad243749c920)) - Julian Hoever
- **(pyproject)** add matplotlib dependency for the qlstm example - ([dadbc20](https://github.com/es-ude/elastic-ai.creator/commit/dadbc20f5e4328d6475c418277b08059b9ba1391)) - Julian Hoever

### Refactoring

- **(creator)** remove unused file - ([ca2b04f](https://github.com/es-ude/elastic-ai.creator/commit/ca2b04f8a3b4204ff18619a89f9fdc44b291b20a)) - Julian Hoever
- **(precomputation)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([2c4212f](https://github.com/es-ude/elastic-ai.creator/commit/2c4212ffa10df0550f5ac0924eee55e46a1ece5d)) - Julian Hoever
- **(qat)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([15f4b8a](https://github.com/es-ude/elastic-ai.creator/commit/15f4b8a52c78a680e3ad95fc70dbd85864282606)) - Julian Hoever
- **(typing)** apply python3.10 typing, remove unused imports - ([f2a31c6](https://github.com/es-ude/elastic-ai.creator/commit/f2a31c6d7d75e1f545ea63cd1ed6f19dc7be7249)) - Julian Hoever
- **(typing)** set correct typings - ([600f6fb](https://github.com/es-ude/elastic-ai.creator/commit/600f6fb9db4e908e7c6eda4652af858258c903aa)) - Julian Hoever
- **(typing)** add missing typing - ([1e58596](https://github.com/es-ude/elastic-ai.creator/commit/1e58596b12eef51de75fe01f60529271f4caaa6b)) - Julian Hoever
- **(vhdl)** remove unused file - ([577c91e](https://github.com/es-ude/elastic-ai.creator/commit/577c91ed4279ed7dbcdae71b5f4e8f868f6092ab)) - Julian Hoever
- **(vhdl)** apply python3.10 typing, renaming functions/classes, remove unused imports - ([1554885](https://github.com/es-ude/elastic-ai.creator/commit/1554885edfe073f5066d82c47182704fdf14d415)) - Julian Hoever
- **(vhdl)** move _int_to_bin_str to ToLogicEncoder class and refactor the class - ([c6495a0](https://github.com/es-ude/elastic-ai.creator/commit/c6495a05c77962ce4cfb4a4110bf0add74d11869)) - Julian Hoever

### Ci

- **(gh-workflow)** use python3.10 for semantic release - ([d7c5b6b](https://github.com/es-ude/elastic-ai.creator/commit/d7c5b6b6fc59ca88532defeb48894a0c792601d6)) - Lukas Einhaus

---
## [0.9.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.8.0..v0.9.0) - 2022-06-22

### Bug Fixes

- **(vhdl)** change value so that it fits into the value range of a fixed point value - ([b4e973e](https://github.com/es-ude/elastic-ai.creator/commit/b4e973ebb8a087351e07966821229f69dc345d79)) - Julian Hoever
- **(vhdl)** remove old brevitas code - ([86a8104](https://github.com/es-ude/elastic-ai.creator/commit/86a8104cb6049dc016a5c8da08a7d2abc011935b)) - Julian Hoever
- **(vhdl)** correct usage of the lookup_table_generator_function according to the type hints - ([9812ee8](https://github.com/es-ude/elastic-ai.creator/commit/9812ee85cd467e261af942b30493ac0e970ea5e4)) - Julian Hoever

### Features

- **(vhdl)** separate hex/bin representation from vhdl hex/bin representation - ([eb8fe60](https://github.com/es-ude/elastic-ai.creator/commit/eb8fe60300ee7572500f9f9d11b62a9c5abff802)) - Julian Hoever
- **(vhdl)** add a function to infer total and frac bits from a sequence of FixedPoint values - ([9cc2b72](https://github.com/es-ude/elastic-ai.creator/commit/9cc2b721b147628b2abf524129eeaac8f68520d5)) - Julian Hoever
- **(vhdl)** change Rom that it uses the FixedPoint datatype - ([876cdb8](https://github.com/es-ude/elastic-ai.creator/commit/876cdb821ff0ac67ae2345c8a36e4a742cce0949)) - Julian Hoever
- **(vhdl)** add function to convert list of float values to a list of FixedPoint objects - ([02b26d8](https://github.com/es-ude/elastic-ai.creator/commit/02b26d868cad2a5a5bed2350a2929cf362ccdca8)) - Julian Hoever
- **(vhdl)** add function to convert a list of ints to a list of FixedPoint objects - ([abece1f](https://github.com/es-ude/elastic-ai.creator/commit/abece1fd38af607c5f5734aeacd77a1743ff3411)) - Julian Hoever
- **(vhdl)** integrate FixedPoint datatype in the LSTM test bench classes - ([7cbb88a](https://github.com/es-ude/elastic-ai.creator/commit/7cbb88a7f77728776e0e976dcc68505b4162f0cc)) - Julian Hoever
- **(vhdl)** verify total bits and frac bits for multiple lists - ([360d318](https://github.com/es-ude/elastic-ai.creator/commit/360d318db0076d9077ceb94f3f7904d95e2b12f6)) - Julian Hoever
- **(vhdl)** add function to convert FixedPoint to signed int representation - ([03001ed](https://github.com/es-ude/elastic-ai.creator/commit/03001ed608ac934e8bbdcdfa1acb2fc7c163a89a)) - Julian Hoever
- **(vhdl)** integrate FixedPoint type - ([b67a609](https://github.com/es-ude/elastic-ai.creator/commit/b67a6096023a51ff4882a8cdd03a7765884c8d93)) - Julian Hoever

### Refactoring

- **(vhdl)** apply python 3.10 typing - ([1c73b26](https://github.com/es-ude/elastic-ai.creator/commit/1c73b265bb8c935d8618f0355c50ca42d1b47168)) - Julian Hoever
- **(vhdl)** small code quality improvement by using the chain function - ([6517cdd](https://github.com/es-ude/elastic-ai.creator/commit/6517cdd4090574d2be5bbbdf6ae68571d6679f05)) - Julian Hoever
- **(vhdl)** use resource_utils instead of importlib directly - ([a7598b4](https://github.com/es-ude/elastic-ai.creator/commit/a7598b4779d98dc0a843f6a90d459f70c6d632f3)) - Julian Hoever
- **(vhdl)** merge gen_func_for_one_lstm_cell and gen_func_for_lstm_layer in one module - ([06158a9](https://github.com/es-ude/elastic-ai.creator/commit/06158a92feca1023dd1b781da691a6965529c842)) - Julian Hoever
- **(vhdl)** remove no longer needed fixed point converter classes - ([4fcf0d1](https://github.com/es-ude/elastic-ai.creator/commit/4fcf0d16cdcac082c19dd654210b5d37991f9139)) - Julian Hoever

---
## [0.8.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.7.0..v0.8.0) - 2022-06-08

### Bug Fixes

- **(gh-workflow)** set more explicit python version - ([9c44093](https://github.com/es-ude/elastic-ai.creator/commit/9c44093c6cd41d05a2d178e6e113bd10f7b86016)) - Julian Hoever
- **(precomputation)** change ModuleProto to Module - ([cfe418e](https://github.com/es-ude/elastic-ai.creator/commit/cfe418e41889708a53c255a8a7abcd6f1648f8f2)) - Julian Hoever
- **(pyproject)** set correct version numbers and add protobuf dependency - ([260e5fb](https://github.com/es-ude/elastic-ai.creator/commit/260e5fb31c425ad9ba2ec31f2fa292961fd28ffa)) - Julian Hoever
- **(pyproject)** correct deps - ([7935ba1](https://github.com/es-ude/elastic-ai.creator/commit/7935ba19bcbda7e47ddbc358c12af3aa2a01df0a)) - Lukas Einhaus
- **(pyproject)** update poetry lock file - ([9230672](https://github.com/es-ude/elastic-ai.creator/commit/92306722dabe5c4196e79a7cbbebab1e75ac3e6d)) - Julian Hoever
- **(pyproject)** update poetry lock file - ([0116934](https://github.com/es-ude/elastic-ai.creator/commit/0116934b994c4e743b1be009172de0e07acd9182)) - Julian Hoever
- **(vhdl)** fix import of Sequence type - ([2c463ac](https://github.com/es-ude/elastic-ai.creator/commit/2c463acdbdae0ed7dc9fa99730f53db94deb7142)) - Julian Hoever
- resolve dependency version conflicts - ([32bd544](https://github.com/es-ude/elastic-ai.creator/commit/32bd544b2e74b8b57497f3fd604deb5ed86ebb42)) - Lukas Einhaus
- fix dependencies + onnx integration tests - ([f06d0f8](https://github.com/es-ude/elastic-ai.creator/commit/f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0)) - Lukas Einhaus
- specify exact python version in github workflow - ([f3ffb18](https://github.com/es-ude/elastic-ai.creator/commit/f3ffb183e86b722cec5efb31e0937c4810542aef)) - Lukas Einhaus

### Features

- **(gh-workflow)** increase python version - ([02403e6](https://github.com/es-ude/elastic-ai.creator/commit/02403e6cb7d8c9acc4357d9649fd2ae0834030a0)) - Julian Hoever
- **(pyproject)** drop brevitas support - ([103f188](https://github.com/es-ude/elastic-ai.creator/commit/103f1882c8da81cdf114f10b1b76c2ce89a07cba)) - Julian Hoever
- bump python version to 3.10 - ([47f5f07](https://github.com/es-ude/elastic-ai.creator/commit/47f5f0718460a966faaa937b2c6b016720434082)) - Lukas Einhaus

### Style

- beautify 6772407f9929e398f7e03858e91b02c52bc8e3ec - ([ecb21e2](https://github.com/es-ude/elastic-ai.creator/commit/ecb21e271271e52d63c268b311d598fb8c86af15)) - github-actions

---
## [0.7.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.6.1..v0.7.0) - 2022-06-05

### Features

- **(vhdl)** start implementing FixedPoint datatype - ([8c4f420](https://github.com/es-ude/elastic-ai.creator/commit/8c4f42097ff416f8e9056af430bda01a5bd42df5)) - Julian Hoever
- **(vhdl)** add rich comparison methods, multiple operators and a bit iterator to FixedPoint - ([116b006](https://github.com/es-ude/elastic-ai.creator/commit/116b00647c05ef6854d3cbd1ab0f79c58f0c450d)) - Julian Hoever

---
## [0.6.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.6.0..v0.6.1) - 2022-05-27

### Bug Fixes

- **(vhdl)** saving generated examples to a directory instead of a giving an explicit file path - ([eb41d8d](https://github.com/es-ude/elastic-ai.creator/commit/eb41d8db9af5171ac2826f41e98b5d85598b582d)) - Julian Hoever

---
## [0.6.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.5.0..v0.6.0) - 2022-05-25

### Bug Fixes

- **(vhdl)** [**breaking**] fix previously broken imports - ([bf694f8](https://github.com/es-ude/elastic-ai.creator/commit/bf694f80fbd3a5478d99e8ae6b198a9e363569c9)) - Lukas Einhaus
- **(vhdl)** move missing files - ([e4ae3c2](https://github.com/es-ude/elastic-ai.creator/commit/e4ae3c2815a33b8f4f33c9578ab5cae0842277aa)) - Lukas Einhaus

### Refactoring

- **(vhdl)** remove usage of protected functions in tests - ([47ca401](https://github.com/es-ude/elastic-ai.creator/commit/47ca401e9c19f3f80140bc9c06c1a3e162c6849c)) - Lukas Einhaus

---
## [0.5.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.2..v0.5.0) - 2022-05-25

### Documentation

- **(readme)** shorten - ([e535ea0](https://github.com/es-ude/elastic-ai.creator/commit/e535ea0fd9d783f29ebb32d756077289d8baa8c9)) - Lukas Einhaus
- **(readme)** fix table of contents and section headers - ([ecdef5d](https://github.com/es-ude/elastic-ai.creator/commit/ecdef5da63c2c10e61a159c144c5c3707a5699e8)) - Lukas Einhaus
- **(readme)** add git commit message scopes - ([fe8e328](https://github.com/es-ude/elastic-ai.creator/commit/fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f)) - Lukas Einhaus

### Features

- **(precomputation, typing)** make IOTable iterable and fix types - ([faa1d77](https://github.com/es-ude/elastic-ai.creator/commit/faa1d7799bd6e8223cc4953170286d425255bb7b)) - Lukas Einhaus
- **(vhdl)** add multiline template expansion - ([309ea35](https://github.com/es-ude/elastic-ai.creator/commit/309ea350fae2b4e54bf06101aadc28e227d30cbb)) - Lukas Einhaus
- **(vhdl)** add multiline template expansion - ([0d7f91f](https://github.com/es-ude/elastic-ai.creator/commit/0d7f91f7347a5501eba02ad40499a5c0fdcce3bc)) - Lukas Einhaus
- **(vhdl)** add multiline template expansion - ([5779708](https://github.com/es-ude/elastic-ai.creator/commit/5779708c2de34d9c32a18af689b4b094d513ef8d)) - Lukas Einhaus
- **(vhdl)** add multiline template expansion - ([3177fcd](https://github.com/es-ude/elastic-ai.creator/commit/3177fcd4e5f1830608e9f6590b5af312bb74b7a9)) - Lukas Einhaus

### Miscellaneous Chores

- **(layers)** remove deprecation warning about Q* layers - ([c696596](https://github.com/es-ude/elastic-ai.creator/commit/c6965961f37a5154356a9b299fc1de36888cd184)) - Lukas Einhaus

### Refactoring

- **(precomputation)** make IOTable grouping an IOTable method - ([c97ec8c](https://github.com/es-ude/elastic-ai.creator/commit/c97ec8c40e1f525a19cdc6838f73be312c209b10)) - Lukas Einhaus
- **(precomputation)** make IOTable grouping an IOTable method - ([9f08625](https://github.com/es-ude/elastic-ai.creator/commit/9f08625704d76776bc7d8f09f15e8571bb81d4ba)) - Lukas Einhaus
- **(typing)** use correct numpy typing - ([3d3ce3f](https://github.com/es-ude/elastic-ai.creator/commit/3d3ce3fe11e96c882e5392cc98ca059addb2b145)) - Lukas Einhaus
- **(typing, mlframework)** move type defs - ([1796473](https://github.com/es-ude/elastic-ai.creator/commit/1796473e2cfb6c0e97c9562844f811878b2b518d)) - Lukas Einhaus
- move implementations to packages corresponding to scopes - ([fa4487b](https://github.com/es-ude/elastic-ai.creator/commit/fa4487b6d491f2f3b089000aca7fe04366b441d0)) - Lukas Einhaus

### Tests

- **(precomputation)** create list from grouped tables to allow subscript - ([0027a3d](https://github.com/es-ude/elastic-ai.creator/commit/0027a3d7faf88d7c2c91f325685953f5fde4e347)) - Lukas Einhaus
- **(vhdl)** start implementation/integration of truth table design - ([40f5396](https://github.com/es-ude/elastic-ai.creator/commit/40f5396f6a207cb72b961a2900dbbefd59dbc5f1)) - Lukas Einhaus
- **(vhdl)** start implementation/integration of truth table design - ([8dc576c](https://github.com/es-ude/elastic-ai.creator/commit/8dc576c9759c770dcdd50f28611ea2082a22f5d7)) - Lukas Einhaus
- **(vhdl)** start implementation/integration of truth table design - ([38f7e15](https://github.com/es-ude/elastic-ai.creator/commit/38f7e15740399350ab34cc42256675482af7d715)) - Lukas Einhaus
- move some test files - ([dc7056c](https://github.com/es-ude/elastic-ai.creator/commit/dc7056c177dff7517d16f3adf5fbfe568eeb85f1)) - Lukas Einhaus

### Ci

- **(gitlab)** remove outdated gitlab ci configs - ([07832c8](https://github.com/es-ude/elastic-ai.creator/commit/07832c85f62e3a71bed507d100685923d70bf424)) - Lukas Einhaus
- **(gitlab)** remove outdated gitlab ci configs - ([68ff755](https://github.com/es-ude/elastic-ai.creator/commit/68ff75501b44027677f320e7b63f83fdfd89ec43)) - Lukas Einhaus
- **(gitlab)** remove outdated gitlab ci configs - ([017cbf6](https://github.com/es-ude/elastic-ai.creator/commit/017cbf67dc8a47f4aa6daea19444388b351f6b97)) - Lukas Einhaus

---
## [0.4.2](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.1..v0.4.2) - 2022-05-24

### Bug Fixes

- **(number-repr)** fix a bug and add some parameter checks - ([a78e9e8](https://github.com/es-ude/elastic-ai.creator/commit/a78e9e8f669c477d0629695f5c7c8ad8628f0522)) - Julian Hoever

### Tests

- **(number-repr)** add tests for the _int_to_bin_str and _int_to_hex_str functions - ([002d7e2](https://github.com/es-ude/elastic-ai.creator/commit/002d7e2cc20d6646973eb343d787392b28d65b26)) - Julian Hoever

---
## [0.4.1](https://github.com/es-ude/elastic-ai.creator/compare/v0.4.0..v0.4.1) - 2022-05-24

### Bug Fixes

- minor errors - ([812809e](https://github.com/es-ude/elastic-ai.creator/commit/812809e1d0e706df3a0514b3503dc283ea12d7a4)) - Chao Qian

### Tests

- fix errors in the intergation test of generating testbench - ([d8378bf](https://github.com/es-ude/elastic-ai.creator/commit/d8378bfa6afaaf84949b284c6b53884d5b5d4ff6)) - Chao Qian

---
## [0.4.0](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.10..v0.4.0) - 2022-05-23

### Bug Fixes

- **(vhdl)** [**breaking**] improve names in scope of ToLogicEncoder - ([67f7312](https://github.com/es-ude/elastic-ai.creator/commit/67f73129faefe343e9fb5e84563d125b1d36bab6)) - Lukas Einhaus

### Features

- **(vhdl)** allow ToLogicEncoder to register symbols in batches - ([9209279](https://github.com/es-ude/elastic-ai.creator/commit/9209279debe651b653d2fee44533ccbdae945b32)) - Lukas Einhaus

---
## [0.3.10](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.9..v0.3.10) - 2022-05-23

### Bug Fixes

- **(types)** add missing mlframework types - ([3b5cf5f](https://github.com/es-ude/elastic-ai.creator/commit/3b5cf5f8be829e109db363c25ecff76634f9d94f)) - Lukas Einhaus
- **(typing)** fix some mypy errors - ([35b8fdf](https://github.com/es-ude/elastic-ai.creator/commit/35b8fdf4cb0736770d9592f86499192e1e84d673)) - Lukas Einhaus

### Miscellaneous Chores

- **(gitignore)** add coverage and onnx outputs - ([a034785](https://github.com/es-ude/elastic-ai.creator/commit/a03478528034243c1cbe8358890bb65a2845423c)) - Lukas Einhaus
- **(gitignore)** add htmlcov produced from coverage - ([5179c0d](https://github.com/es-ude/elastic-ai.creator/commit/5179c0d526dd549fe06101342a33f89117acc022)) - Lukas Einhaus

### Tests

- **(brevitas)** move brevitas tests to integration tests - ([7ba7757](https://github.com/es-ude/elastic-ai.creator/commit/7ba7757e21245f6c418b5c12f5e3d1cc0bee9a7e)) - Lukas Einhaus

### Ci

- **(gh-workflow)** ignore main branch for checks - ([56a6640](https://github.com/es-ude/elastic-ai.creator/commit/56a6640d9839447880ca6b8e6ca495615bc86454)) - Lukas Einhaus
- **(gh-workflow)** use pypi auth token instead of testpypi - ([fa2faae](https://github.com/es-ude/elastic-ai.creator/commit/fa2faae0e9fc4dc09b9de9f8b9c5032dfc104ecb)) - Lukas Einhaus
- **(gh-workflow)** remove main branch from pull_request branch-ignore filter for checks.yml - ([c081fd9](https://github.com/es-ude/elastic-ai.creator/commit/c081fd984f3f8ea8bfa3bc4552980608d73badc3)) - Lukas Einhaus
- **(gh-workflow)** publish release on github - ([7c0fb1c](https://github.com/es-ude/elastic-ai.creator/commit/7c0fb1cc74956257ce3fa93288c7f4dffacfef54)) - Lukas Einhaus

---
## [0.3.9](https://github.com/es-ude/elastic-ai.creator/compare/v0.3.8..v0.3.9) - 2022-05-22

### Miscellaneous Chores

- **(pyproject)** update version number - ([94bdcab](https://github.com/es-ude/elastic-ai.creator/commit/94bdcabfa74fe4e634932eac6e4b5e36a02df236)) - Lukas Einhaus

### Ci

- **(gh-workflow)** enable test-and-publish for main - ([78870c1](https://github.com/es-ude/elastic-ai.creator/commit/78870c1400b72c7d847b3da2d346f8eaf14fd619)) - Lukas Einhaus

---
## [0.3.8](https://github.com/es-ude/elastic-ai.creator/compare/0.3.5-alpha.0..v0.3.8) - 2022-05-20

### Bug Fixes

- **(gh-workflow)** fix syntax error - ([895326d](https://github.com/es-ude/elastic-ai.creator/commit/895326d67eb7ba1bb866a45c8b149778c93dc043)) - Lukas Einhaus
- **(gh-workflow)** bump version - ([2cb3a72](https://github.com/es-ude/elastic-ai.creator/commit/2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6)) - Lukas Einhaus
- **(gh-workflow)** close brace - ([a6e4b99](https://github.com/es-ude/elastic-ai.creator/commit/a6e4b999dadd163881fa96d03977c9c392a9267b)) - Lukas Einhaus
- **(gh-workflow)** typo unit-test -> unit-tests - ([1dbd71f](https://github.com/es-ude/elastic-ai.creator/commit/1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73)) - Lukas Einhaus
- **(gh-workflow)** fix typo - ([7f86205](https://github.com/es-ude/elastic-ai.creator/commit/7f8620502ee544917db42ea12c7cb2eadbaef8cc)) - Lukas Einhaus
- **(gh-workflow)** fix job dependencies - ([6a7d3ee](https://github.com/es-ude/elastic-ai.creator/commit/6a7d3eeb975ca303aa30fce21bd29d14cf9982d3)) - Lukas Einhaus
- **(gh-workflow)** set git user+mail to gh-actions - ([174ed47](https://github.com/es-ude/elastic-ai.creator/commit/174ed478b04b846912d6b0315f1143f24bc94524)) - Lukas Einhaus
- **(gh-workflow)** install latest isort fixing broken imports - ([a61ef44](https://github.com/es-ude/elastic-ai.creator/commit/a61ef445672f913ec4ebc4cc8b46c2ef9099bec7)) - Lukas Einhaus
- **(gh-workflow)** updat changelog correctly - ([e76a41c](https://github.com/es-ude/elastic-ai.creator/commit/e76a41cf55463cbc2a4ffa5b2b233d49695302b9)) - Lukas Einhaus
- **(input_domains)** add missing import of itertools - ([a6b0344](https://github.com/es-ude/elastic-ai.creator/commit/a6b0344ac4b933112b19b8603358a0adc7274533)) - Lukas Einhaus
- **(pre-commit)** add missing commitlint.config.js - ([2251de8](https://github.com/es-ude/elastic-ai.creator/commit/2251de83f60823d21346aedcc2b2e9aac4c27458)) - Lukas Einhaus
- **(precomputation)** correct numpy namespace for type alias - ([a6c5842](https://github.com/es-ude/elastic-ai.creator/commit/a6c5842920c00ae6e53e226650e0fbfe48aac44a)) - Lukas Einhaus
- **(pyproject)** fix duplicate field - ([6616cab](https://github.com/es-ude/elastic-ai.creator/commit/6616cab3b0342f0b5d0b8bbdbbdf719de56d5631)) - Lukas Einhaus
- add missing tags_utils again - ([910c611](https://github.com/es-ude/elastic-ai.creator/commit/910c6116600b82e2c52c7d46896d92b63954d7c7)) - Lukas Einhaus

### Documentation

- **(readme)** add brief explanation of pre-commit - ([3626bb0](https://github.com/es-ude/elastic-ai.creator/commit/3626bb07cc1c8600b193bc380ae8275116ebaba8)) - Lukas Einhaus
- update changelog - ([e1aa8c9](https://github.com/es-ude/elastic-ai.creator/commit/e1aa8c93554fc15c25a586b8e89eecda6dc03514)) - github-actions

### Miscellaneous Chores

- **(gh-workflow)** setup semantic versioning - ([ac06cf2](https://github.com/es-ude/elastic-ai.creator/commit/ac06cf26b4cfd66a3b49b82106ace1f236c01eb4)) - Lukas Einhaus
- **(gh-workflow)** setup semantic versioning - ([f714503](https://github.com/es-ude/elastic-ai.creator/commit/f714503474da4695a396ca784f110378b2147591)) - Lukas Einhaus
- **(gh-workflow)** use emojis for automatic semver - ([93a60cc](https://github.com/es-ude/elastic-ai.creator/commit/93a60cc2755c08098b9c1a1f8ff5dfecee289c76)) - Lukas Einhaus
- **(gh-workflow)** remove tag_utils.py - ([a8baca4](https://github.com/es-ude/elastic-ai.creator/commit/a8baca48073d6efa1330f82f87f23ec205ac02e9)) - Lukas Einhaus
- **(gh-workflow)** automatically update changelog - ([45bfef3](https://github.com/es-ude/elastic-ai.creator/commit/45bfef38bd0dc3e86a9a291553b1f3ea5570dc9e)) - Lukas Einhaus
- **(gh-workflow)** deploy to pypi instead of testpypi - ([18aee87](https://github.com/es-ude/elastic-ai.creator/commit/18aee872212ba9f066d579e4c2a5edd11e5b4a59)) - Lukas Einhaus
- **(gh-workflow)** manually trigger precommit workflow - ([f6611d9](https://github.com/es-ude/elastic-ai.creator/commit/f6611d9360f2b8a9ece7ace714050c85884fd6ce)) - Lukas Einhaus
- **(gh-workflow)** trigger precommit on push - ([391cc8e](https://github.com/es-ude/elastic-ai.creator/commit/391cc8ef81c2d92bf432adb7814cbe95e9961c38)) - Lukas Einhaus
- **(gh-workflow)** correct token for  test pypi - ([112eb37](https://github.com/es-ude/elastic-ai.creator/commit/112eb374c0f4b43b61b4988e8425a82881bd6802)) - Lukas Einhaus
- **(gh-workflow)** remove pre-commit usage - ([9cd3f34](https://github.com/es-ude/elastic-ai.creator/commit/9cd3f34c8e8b6ef1dc0904f071b1d2e3a2c0e684)) - Lukas Einhaus
- **(gh-workflow,pyproject)** build via semantic-release - ([e78e882](https://github.com/es-ude/elastic-ai.creator/commit/e78e882bb02b0a7adc9ff10c437a37bf6cc08dbc)) - Lukas Einhaus
- **(gh-workflows)** try semantic release - ([8a23dbf](https://github.com/es-ude/elastic-ai.creator/commit/8a23dbfeafeae82f1332c3cd28c5cbf72215a9c8)) - Lukas Einhaus
- **(gitignore)** add node_modules - ([23e1234](https://github.com/es-ude/elastic-ai.creator/commit/23e12348b598edea69cf0a79e4bee26c45f62f43)) - Lukas Einhaus
- **(gitignore)** add mypy cache - ([0a8c31e](https://github.com/es-ude/elastic-ai.creator/commit/0a8c31e0045ad244192bdb9fc91803a5d6470de1)) - Lukas Einhaus
- **(gitignore)** add npm files - ([352c38f](https://github.com/es-ude/elastic-ai.creator/commit/352c38f3c83982b3abd52eb0d2bb1a654ff9bb57)) - Lukas Einhaus
- **(pre-commit)** default install commit-msg stage - ([28c4716](https://github.com/es-ude/elastic-ai.creator/commit/28c4716672a446623ad957b5f7f090f1eff211af)) - Lukas Einhaus
- **(pre-commit)** put mypy+dead into manual stage - ([98d9620](https://github.com/es-ude/elastic-ai.creator/commit/98d9620f3a33a42a14d3dae04841660e82187609)) - Lukas Einhaus
- **(precommit)** configure hook stages - ([1d9ffc5](https://github.com/es-ude/elastic-ai.creator/commit/1d9ffc57bdad9832c286d70412a2dcccce866f29)) - Lukas Einhaus
- **(pyproject, gh-workflow)** upload to github - ([cd1a8d5](https://github.com/es-ude/elastic-ai.creator/commit/cd1a8d5a14a462db4bde32b769f88fa15aebaebc)) - Lukas Einhaus
- **(semver)** revert to angular style commit messages - ([55f99dd](https://github.com/es-ude/elastic-ai.creator/commit/55f99ddd6f809169f91d707a51f29477523f26b0)) - Lukas Einhaus

### Style

- **(imports)** sort imports - ([de31b33](https://github.com/es-ude/elastic-ai.creator/commit/de31b335ed9ee8cf04d3823d0b9058e54df07eb9)) - Lukas Einhaus
- beautify 6fe04eccb8dc55714b78e1a7222113c93a0b258c - ([919ac6e](https://github.com/es-ude/elastic-ai.creator/commit/919ac6ecfc5702c9a705f3da181916c2b9265366)) - github-actions
- beautify 1d617cd289068f3c6552da1bd6e9468759cb5747 - ([0bb5d39](https://github.com/es-ude/elastic-ai.creator/commit/0bb5d39e73c6b4e746f1fb0308b863273d86b7f3)) - github-actions
- run pre-commit tools on all files - ([c22eecf](https://github.com/es-ude/elastic-ai.creator/commit/c22eecf97792e104596e6575d692c6f4564e66c2)) - Lukas Einhaus

### Ci

- **(gh-workflow)** move unit/integration tests to checks.yml - ([a3ffe51](https://github.com/es-ude/elastic-ai.creator/commit/a3ffe5137af975d516d00240f1517f58fbed9196)) - Lukas Einhaus
- **(gh-workflow)** disable mypy typechecking - ([f63ded1](https://github.com/es-ude/elastic-ai.creator/commit/f63ded1fa71a9137d9e2502e7e9b693682116302)) - Lukas Einhaus
- **(pre-commit)** use pre-commit in checks - ([c3b741a](https://github.com/es-ude/elastic-ai.creator/commit/c3b741a9554dd63ec02ee1044a96bfd2fa658bfe)) - Lukas Einhaus
- add test publish workflow - ([473e533](https://github.com/es-ude/elastic-ai.creator/commit/473e533dd8046e922416e121e60971a9ce2e9641)) - Lukas Einhaus
- update repository url to test pypi - ([fab7dbb](https://github.com/es-ude/elastic-ai.creator/commit/fab7dbba3bd800189b04e1f13434440b6b1be603)) - Lukas Einhaus
- add next version placeholder to CHANGELOG.md - ([2c94335](https://github.com/es-ude/elastic-ai.creator/commit/2c94335e57dcac4cdfa1d612a4858e5fa0c34a8b)) - Lukas Einhaus

---
## [0.3.0] - 2021-12-15

### Wip

- start tags feature #174 - ([31d62d2](https://github.com/es-ude/elastic-ai.creator/commit/31d62d23c9a10bff23ee2ff8fc10f443218a8be3)) - Lukas Einhaus

