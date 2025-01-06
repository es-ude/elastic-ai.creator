## 0.60.0 - 2024-12-18

### Build

- Add missing pytest-cov ([1707f07](1707f07eb91693a03e3e33e1f3a4cc1ab12037c2))
- Clean up pyproject ([fa27806](fa27806b53b5d27f84c4ccee57e7abefbe242cf6))
- Omit *_test in main folder for coverage ([da8104d](da8104da6da54438104cd6bdecd68cc06d08cadd))

### Bump

- 0.59.2 -> 0.60.0 ([1eb0ade](1eb0ade5030776dc820d3fedb295d07536119242))

### Chore

- Only throw a warning if commit message exceeds char limit ([3e1d509](3e1d509e23d8aa5302e708bb309514294b9d7984)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Use python3.10 to run the tests ([3151ba2](3151ba269b6265c1f90bf12b125f0c62e5e969f0)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Synchronize pyproject.toml and poetry.lock ([76a8baa](76a8baa01b52daf59248c33994e747f7b3dc4cb8))
- Added package.json to fix the verison of commitlint ([f8a7a0f](f8a7a0f3f888e4937baa8d0c7423637facaf443d))
- Fixed dependency of elasticai-runtime-env5 from develop branch to specific commit ([a5ac0df](a5ac0dfb7e0db20f21daaa75d4dd1e162f298cea))
- Update pre-commit (necessary to fix broken black deps) ([9102843](9102843a898829a0dea0001f009a41265b4cf919))
- Add devenv/direnv/uvlock ([476cefa](476cefa326eb3084a74717c872981dbbf97feff0))
- Create coverage report for develop as well ([8c11c01](8c11c01ae66485d70538f07effbb637533de085f))
- Clean up external deps ([2f83d46](2f83d46e0a39a079f9f3884ee71eb37a87d972c0))
- Run black with python3.12 to support new syntax ([ff51308](ff5130872f82206e68075f9b6c99b53dfa746a39))
- Remove redundant tests ([3f0c243](3f0c243ad1f35172e199e94d3fe060b86b943661))
- Add test/build/style as types ([7d00767](7d0076794361fc99aeaaf7b80474679e8cd6d257))

### Ci

- Publish on pushing release tags ([4b16c98](4b16c98ed20599385e49a49e44a995cf4ea73dc6))
- Install only testing group for unit-tests job ([73b3bf8](73b3bf822e298fde23a045d4297ff3ca48773383))
- Perform coverage when running checks ([75b520f](75b520f300e4ac0bd36ed08c7c92ace70d44c64c))
- Don't install lsp deps in test pipeline ([07853aa](07853aa973d5e290d869b96c3168ed5bda1cde7d))
- Remove publishing to test.pypi.org ([44d50d4](44d50d446b8bc6a1f41a563fc3cb52ad61bf04fe))

### Docs

- Add register documentation ([59f7ed4](59f7ed4c044b6062d37d66c7dc97cb31a056939b))
- Add more middleware/skeleton specification ([b62d982](b62d982f13360adffdfbd6a041ae30aaf83f7571))
- Add timing diagram to skeleton/middleware spec ([574116b](574116b529b20334c6646de1fd20f3e95dc47218))
- Explain we need to read each result byte two times ([96572fb](96572fb21958c7b79505e1ea004cdf9681e8097d))
- Fix hw function id length ([3539c9f](3539c9f65a6bccf72c4cfb0312a4b2408e0b4fb9))
- Removed unnecessary comments ([25cdf90](25cdf904571da8d4f60418ce52447c8959b4c87b))
- Added comments to parsing functions in testbenches ([55c9f4d](55c9f4de6ce269f60f25edd91592aea1debe8701))
- Added more context for the parse reported content functions ([70c8b4b](70c8b4bbd01aaf272ccf6a91af4d91a333dce41f))

### Feat

- Added a bash script to automatically build the vivado file with the help of vivado 2021.1 on a server ([eb8c835](eb8c835529d736037b085f5ede0490ca342bac3e))
- Set more specific return type for create_testbench function ([7e3f54b](7e3f54b5cbc7aa6d026d60b28f8c32c05768aa0c))
- Add general skeleton class ([b4ffacb](b4ffacb1847685851def6beb9f53044fe5dbd75f))
- New firmware that does not save testbenches ([8ad3272](8ad3272c80a350348df1ce7a562df6f51928a4ee))
- Test that firmware generates skeleton correctly ([3a18656](3a1865642e60fcd4f4fbf49917d0930663bc38aa))
- Create separate LSTMFirmwareENv5 ([17a274c](17a274c6bb23fb5721b3e07cd16916bcbd3889c8))
- Add skeleton for sequential layer ([34e8202](34e8202281e0be4d79d78df66cbcffc9b4db3878))
- Add support for less than 8 bit in skeleton ([231f0ca](231f0ca808248b740421e5bb516b71e5f0c434ce))
- Convert negative numbers to bit patterns using two's complement ([c94dc3b](c94dc3ba59698e87eac2efe702a43dfc925401bd))
- Added skeleton version 2 to project ([6ed2c94](6ed2c94abbcb0d090eac3844fb59e27983f7ed11))
- Remove restriction to pytorch versions < 2.0.1 ([bb47705](bb477058440e07e2bdd6c467e328219519510771)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Allow python3.10 ([0628024](0628024ba826ebbdbe5b5deda4aac67d81876248)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Added an basic example for a network using skeleton v2 ([6d94158](6d941584c40d41049bd27e9da8c2dc204f79080b))
- Allow '${key}' placeholders for multiline templates ([d25eef1](d25eef1369c754911e56ed5aa4a92f62b2716325)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Allow python3.12 ([46d6cfb](46d6cfb52eb2c6de471d9dd0310a9152200ec0db))
- Abstract class DesignCreator inherits from torch.nn.Module ([d6e70ed](d6e70ed1c84d025bb2eebbfda45c67ad9ba1f987))
- Added conv1d. Simulation works. End-to-end system test is still pending ([93e5ecd](93e5ecdaae2987d34772a504bc843019419bd845))
- Added simulation for linear layer ([aac395b](aac395b9413943d6252bf5cb4866d96173216ae8))
- Added enV5 usb library to development pyproject.toml. This will be used in the future to do system tests ([3089341](3089341849008fbfb1ff66029ea522702cc4303f))
- Added a generator for echo server with skeleton #378 ([2c3faf5](2c3faf575df21f1aca236138257097ddd2320bff))
- Added an example for the echoserver with skeleton v2 ([3f46780](3f46780d44dc6f0d220b3a3d82f71e33ae38fdac))
- Echo server works now ([a4359a0](a4359a0f08fa5a620ce414c8de6e133613427a65))
- Linear layer system test with elastic node works now ([238964a](238964a119b31db57c44085c336c8605e10c8e9a)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Add graph delegate and iterators ([868a188](868a188aa4ce0be92608997bbfdb2916e7f8603e))
- Add abstract ir data class and nodes ([bb81f0d](bb81f0dea0ee8ea8646e57d85cc070baddf91e8a))
- Added fixed point config, autograd and quantize ([97bb203](97bb203898e6d689fff54c73da722584aca6882f)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Added basic layers ([82db217](82db217242fed30a955dcf7a69eb98a56e4b931a)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Introduce read only field for IrData ([be1e8fb](be1e8fb3a659d30882b77399016fa8c21d8f0e6b)), IMPORTANT:Changes are still possible via the
`data` member
- Make suc/predecessors sorted/deterministic ([8825607](88256079288bae95668ffe60aced222e920419c3))
- Add basic graph data structure ([e1dfef2](e1dfef26688ffc819fe07ae7831df6b899c4b7f3))
- Add basic function registry ([3785db0](3785db0d42f2f7a9118b9b5c3e60d1f83f9bbd86))
- Add LoweringPass class ([d598021](d5980214c2441b52eef063bed865de2eecd52f10))
- Add automatic deterministic skeleton id generation ([eb7e59f](eb7e59f7aa7506206e6807ef6a649e8b458930b4))
- Tweak api for skel id computation ([f7d9a77](f7d9a7786e06a500dafcc6cbf3f08f81083c6166))
- Move hw accel meta to dedicated module ([9f65b8d](9f65b8dce91fcdfd87f7f1229a06c2d3776f8ad5))
- Load plugin description from package ([05a99c3](05a99c3e71a3c408a4ab273b7fe453b215d39ef9))
- Load plugin and call generated fn ([0492e0b](0492e0b94eae88eb536bd7b85859527026ec273d))
- Move plugin_loader and type_handler decorators ([6bba61d](6bba61d5f7758f1b9db14a0938e29f4c163c52b9))
- Add plugin loader and improve function registry ([0a8ac61](0a8ac61fef8792ab177f7635d86d4f9ae23029b1))
- Add basic but flexible templating component ([2ae0506](2ae050611b9bc2cc93624e99bad7c1244dd2b6c4))
- Remove ir2vhdl (shouldnt have been committed) ([20fb891](20fb8916f78ea3a78ea7eeef9af1d3f071168ca2))

### Fix

- Save testbench to separate folder ([9937431](99374317966a0db05c147bf99d322da5b14b0f5a))
- Xil to work lib ([005ed36](005ed36a4ff8bac6bb1ba1ed29e5e9cfe0be6c73))
- Fix tables and language ([63a1b9d](63a1b9d42aa7a8f3866b978516a7269cec10e61b))
- Fix wrong signal name in integration test ([74ebc32](74ebc32e938936d2d60c3da09773439c5675106d))
- Added skeleton_1.vhd needs to be changed ([632bf89](632bf8974ace775c8289351d03c026e587c237ed))
- Add expected newline to end of skeleton ([dcb20b7](dcb20b712945439b1c3db799404beb33d8587e4f))
- Fix skeleton for mlp use case ([e4b67cc](e4b67ccbc4178629f35f3f3a89259d2bfae3aba0))
- Transmit high byte first instead of low ([f2bd5af](f2bd5af1cf6d9e4f40da8c89a89c61663cf12086))
- Correct counter in example code ([1860657](1860657968f0828502eecfb892c6e58fab93bf10))
- Update deps to resolve security issues ([6568d28](6568d2830120922f77d2e183aa5764369143135f))
- Fixed the test for the old skeleton and added another one for skeleton v2 ([87b11a4](87b11a4455ec059ed8e3fdb3a405a976435facd6))
- Added an exception raise for the skeleton for not supported configurations ([11d006c](11d006c146b9eaa809460729132222a0595e6793))
- Warn when using skeleton v1 ([5a46331](5a4633193cf2c6db699fa19c47ddfbc53599c1fe))
- Implement function only supported in python3.11 and higher ([5223adf](5223adfdf6551f9758ee4dbdef9df1f2aed36377)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Fix incorrectly sorted inputs, add input/output widths and assert them ([74f5e26](74f5e265fb388249a3d46c1743bc3b3e38366a78)), Signed-off-by:Julian Hoever <julianhoever@posteo.de>
- Fix wrong path that leads to temporary files created in the project tree ([0ff9c0b](0ff9c0b04442e6da0d76f3991254cae63bf260e8))
- #374 remove unnecessary data_buf from top module. It is not used anywhere so should have no effect ([e8cd5a7](e8cd5a797c92b4f29b94dad1a9b7de4d090a98ae))
- Fixed fxp_mac reset. This modules reset is now working at any point in time independent of the other inputs ([054b8ab](054b8ab4d3569b3ae105b791ea0e8f116a8ddfd6))
- Linear layer uses signals now. So simulation works ([7f5d30c](7f5d30c6c066506b66112c9ba15fe367ce33f9a8))
- Fixed the dependency for the runtime utils ([cfaf318](cfaf318915a046f0b5707a56c2fcbdb9e312f1dc))
- Fixed the poetry lock file ([31868ca](31868caefb3959d0966c93d01a45c234a9041b55))
- Revert changes in linear.tpl.vhd ([f11b606](f11b6061ff88c4b16e70131508b9f10758c9b90d)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Fixed error in template ([948af39](948af39669a9070518a96bd3611cb6d43b405986))
- Fixed an error in the skeleton v2 template ([dc2ee96](dc2ee964c8898a359fafe75b2bcb216ab39ebf2a))
- Fixed the test for the firmware with skelton v2 ([75ef96a](75ef96a21236c9fc3a6e830aedc5978a9e033c9e))
- Fixed error in test ([2d7f140](2d7f140bc9382be47074c1cbda3015f10ecdfaab))
- Fixed error in convolution ([390656c](390656cc00cd4f827c27badae232ac8073f480a2)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Fixed test for changes in sensitivity list and for rising/falling edge clock ([088bc1f](088bc1f7a206739701d6bf9735b3974add0262c0))
- Fixed code generation test for linear layer ([7f8f445](7f8f4455ea1352d0c30fea05e22fb7ce561d654c))
- Fix bug where iterator was not remembering visited nodes ([3da0dbc](3da0dbc5e84748fdd7db5ee78b9cd40636f19e7e))
- Fix conceptual problems with abstract ir data type ([1e6210d](1e6210db742f3e1b9b2613126cc48262e6eddee4))
- Remove dead code and fix typing ([609eb51](609eb51c45c298e190a1e6f2133623b456e9ee2c))
- Make graph iterators deterministic ([2c3b27a](2c3b27a0e8afbf7bdbea3ce8e45abbbc65408184))
- Removed init in elasticAi ([be65e7c](be65e7c223e339b5ec03fc3b54ec3e4782a58d98)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Remove toplevel __init__.py ([c7f0a78](c7f0a7820c094789d8ae7e4bc9076c5cda167f8d)), IMPORTANT:adding an __init__.py to the toplevel
directory prevents to use elasticai as a namespace
package.
- Make type hints 3.10 compatible ([db8a0f8](db8a0f8dc836e09bc4cc978e574b4d22be798954))
- Remove outdated srcs in skeleton plugin ([57ae044](57ae0442099fe36a2e8c31fe100d2eba59779093))

### Refactor

- Rename _integration_test to example ([6828739](6828739ea5ee27cbe077b5c31ffbf14d66d5f480))
- Remove unnecessary files ([26b8afa](26b8afaa457969777b03f04e537470f1f6917055))
- Added __init__.py ([ad897c2](ad897c20df13c550eb8110299c2c85f3ba960eeb))
- Move design_creator module to nn package ([f17a6da](f17a6dac5e612ba99bfc906862de3e3048aa7a17))
- Rename the module design_creator to design_creator_module ([49d3ab4](49d3ab4e1d244c13d4994df65dda2f47a846aaad))
- Better ghdl simulation class ([873fd42](873fd421db4f0bfb179479c43ee459b71dbeee01))
- Made the name a property so the name is already set correctly and still accessible ([593682b](593682b933642a809496dd2a9b00fdce0e9ba19d))
- Changing the wake_up signal to best practice method ([04221ec](04221ec93724939dcc3bc4d3e28428ca1afffe28)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Making the test a bit more convinient ([6c778bc](6c778bca80ed20c85f2ed2d00701e3b0f2152486)), Signed-off-by:Leo Buron <leo.buron@uni-due.de>
- Add new replacement variable in log2 calculation of linear layer ([082b8fd](082b8fd23ba37f7428b49ab7dcbfabb056c37544))
- Changed sensitivity list to clock only ([01dd3c5](01dd3c5095d2d515cd1516b3a7ca56fe370bee6d))
- Moved the opening of the serial port to context manager ([8431bc7](8431bc74cd1f8444df5a0b6f5b184e52979ffc95))
- Moved mac operators to vhdl shared design ([4925d67](4925d673e8086fceb5af31d1e577c56a003f1dd2))
- Moved simulated layer. MAC operator design simulations do not work ([3b927b6](3b927b693e989a4e82f97b925df28641b7b33fab))
- Removed unnecessary print statements and added type hint ([c3615bd](c3615bddd1a0fab13001dc2457b5f9094c7a91e7))
- Avoid keeping two registries ([21a951e](21a951ed97229f2451d4ddf28c52db794e6f86be))
- Decouple fn registering and calling ([ab737b9](ab737b9bc4f86781e45d5d0b2d804ab5892a495d))
- Use new descriptor for registering fns in lowerable ([2bd382d](2bd382ddd78212aaf925592b9c7f7838c85e89cb))

### Style

- Beautify commit # [9035010](https://github.com/es-ude/elastic-ai.creator/commit/903501083d6acaee8b472f22e7bf24cddb3647b8) ([cc6fda5](cc6fda534289fdf4359c46e9e08ba984c7638a07))
- Beautify commit # [b61c618](https://github.com/es-ude/elastic-ai.creator/commit/b61c6180a56de314e569338eebbbdbe45a889f42) ([ce92f2b](ce92f2b593d3b705e34867a8b87d90e3f4a7d9a9))

## 0.59.2 - 2023-10-06

[9e690eb](9e690eb35378b3915fe8c12226fc12e2b64974c1)...[a8aaa00](a8aaa00ce80b9ffc2e3684648a63c47759974023)

### Fix

- Copy model to cpu for quantized inference ([0c5d88e](0c5d88e26e55eb11d2a729c5a7bf6b865927b61f))

## 0.59.1 - 2023-10-06

[3314acf](3314acfafe001683a31ad211bcb9599c54acb886)...[9e690eb](9e690eb35378b3915fe8c12226fc12e2b64974c1)

### Docs

- Explain relationship between LSTM, LSTMNetwork and their sw/hw impl ([7db8974](7db8974e11a586e60c7f154e7cbbbf27b75a9c41))

### Feat

- Reintegrate lstm implementation ([9440fbb](9440fbb1ed9c81218e62f4e60917127e128d3856))
- Added skeleton, middleware, top module and Vivado constraints fopr env5 LSTM example ([67117a4](67117a443032cbafd3bcf8abcab7177a801fd659))
- Add lstm_network testbench ([37d7921](37d79212f52b949634f7af321e7b7bc56306ffeb))
- Inject network to FirmwareENv5 ([bf2c53f](bf2c53f554bfc312875f08cb99d95e028364667b))

### Fix

- Turn `vhdl.top` into a package ([2a272ea](2a272ea4e0b470bf8684098cf48e8121ee92d27f))
- Add saving constraints and sources to plug&play ENV5 ([41aa4f1](41aa4f105fb59bcdc76b031800af916eb0c76f35))
- Added missing file for network skeleton tpl ([38fe8a7](38fe8a7337479e5b7c02d66d682431e95b03e190))
- Parametrize names ([c13576d](c13576d380c58f658c4054a645f8407041a95faf))
- Fix lstm names ([da279f7](da279f742409b9c31d690d385c4d04896e3afbb4))
- Fix lstm test bench file name ([f879b4d](f879b4de526bc7c7cd93371b5874c1eac2f465f5))
- Correct `create_testbench` for lstm ([e82af52](e82af52217f0ef1874abfcd0b43f1d905ed3e4bb))
- Move `create_testbench` to correct class ([53bc568](53bc5684682abbc721969f357b0810175a89a25f))
- Names and templates for lstm ([3ad358c](3ad358c698bca447376b910bdd275bc806eb6db6))
- Remove unnecessary instance name templ variables ([7da4b5a](7da4b5a473e9976392eda1d4e686cc4ff9b12d0d))
- Don't save uut in testbench ([7f09a2a](7f09a2ab3549d309759e531dd5b6ec4051a9d3e7))
- Add skeleton, etc. to generated files ([2bbd588](2bbd588ceaec6408f61f48c2f61289c90afffef9))
- Fix fxp mac test ([baad73b](baad73b8e97f5c7b65e52a1c9755eb2086de02aa))
- Fix skeleton test ([12b7c27](12b7c274d1d9116d06be711066f2d5ee1cf5725e))
- Use linear layer name ([9d48f09](9d48f098e60ead2854af61a6c1394e824e762538))
- Skeleton naming ([21d057d](21d057d2497d817c09c12e90f43047aeed71e6d8))
- Set step_lut as non trainable parameter ([95954c2](95954c2cbd52a85f762128ce3a88259085431536))
- Do not copy input to cpu ([586b774](586b77458e9529dc8f12023fbefb6a3747fd222e))
- Add step_lut to state_dict ([e18f46f](e18f46f0a70b1000e8e6d0ea3ecdddce2ad325d5))

## 0.59.0 - 2023-10-02

[5e4f666](5e4f6666f6be1c68f7b99bb9f10c055ab96a80e5)...[3314acf](3314acfafe001683a31ad211bcb9599c54acb886)

### Chore

- Add simulation test tag ([f7fda58](f7fda587861e83c729eaa90791b34efc4f9b433d))

### Docs

- Add documentation for number conversion ([617e2c6](617e2c61fab2b21459a1a62c65c184cdbcc35e09))

### Feat

- Lstm reintegration ([0ffa9b0](0ffa9b029636ab36570019e9f99fd5c788281e26))
- Add number_conversion ([1fde323](1fde32308a83a3de71af736eda47a81bb5bc468b))
- Parse ghdl output ([3f7e05f](3f7e05f44f86776ea01b28b176116730d94a9354))
- Handle colons in ghdl sim parsing ([f485531](f485531881b0874eb14df58ccb3873889cb1cac6))
- Basic fxp mac + hw/sw simulation tests ([f34a1ed](f34a1edc70c34d456daa222bed537d793ee0c29e))
- Add xnor-popcount based mac bin impl ([6a63eb3](6a63eb358ce3279bcdbca468aed25445ab0be13e))

### Fix

- Remove need for csv in testbench ([f905bbf](f905bbf6e1f9119677b7f73caf58b35343e0d7bb))
- Ignore one more line for ghdl out parsing ([c48ec8f](c48ec8f42f906222223b7282c677ffdfe5cd06ec))
- Exclude simulations from coverage ([3b31a45](3b31a45bccb262d60a1d920618457514a2bd8a95))
- Make mac impl use round to zero logic ([f8c674f](f8c674f5a2382a1d8b9b1630a8af35977dc0c9c9))

### Refactor

- Simplify number conversion implementations ([715d6c7](715d6c7540748fb53e93f78782f653ee45e6bdb4))
- Simplify simulating test benches ([b607211](b6072118e1d84e59b1faac73ec3c75c6aca88ee9))
- Rename hw_integ_test.py ([8324d1a](8324d1abaad92a12d5fcdee9925c58b9c9743aff))
- Move number conversion modules ([ed3086c](ed3086c149388876e8cd243cd535199bded8f9f5))
- Add create_simulation method to layer ([6c93c81](6c93c81b5910dfcc2eb71d64c137be5f9b8d0fad))

### Style

- Beautify commit # [f94e16f](https://github.com/es-ude/elastic-ai.creator/commit/f94e16fd03d289124dd20dd844776d517fb91e4a) ([db35406](db354066168885e5ae91d18efd705d239319b81a))

## 0.58.0 - 2023-09-29

[a5aecb3](a5aecb3cbc65e1231d1e2de957f28a1fdaba427c)...[5e4f666](5e4f6666f6be1c68f7b99bb9f10c055ab96a80e5)

### Docs

- Use create_design function instead of translate in minimal example ([b9351ca](b9351ca28b37a0ee7f34187b01986c3ba11c6827))

### Feat

- Add tests for conv1d layer ([c645297](c6452975ef02d6c2a46ca4550311238a46917636))
- Add parameter getters ([3c184c0](3c184c075cdb94ffae05ea7424e33dd98a4c09f9))
- Add tests for the conv1d design ([c2c94bd](c2c94bd91e09e507120cfcf71e28b0797dca419c))
- Use bias as default ([14b01be](14b01be0e3aad2315daa1864588701ce2fd8dff7))
- Add tests for fixed point linear layer ([2e959b6](2e959b619ea9ef66dfede042131f7116b18f3532))
- Add tests for linear design ([55a366a](55a366a34049708ee62d0c440f67640435716900))
- Add small integration test to verify that linear layer creates correct design ([f71e43f](f71e43f92cbfbf2c7a735a610c3dbca790ea8299))
- Add small integration test to verify that conv1d layer generates correct design ([da45cc3](da45cc30b481ee27e9bd9d5d172e29a0bd0b519f))

### Fix

- Remove unnecessary output quantization of the SiLU base module ([350faa5](350faa52d93994738a68cc0222dfb907b5174f12))
- Remove stride and padding as supported parameters for conv1d ([05b57d1](05b57d1d2e112a21259fa2221f2204b3f6d87bfe))
- Remove already dropped padding, stride and dilation ([79494fe](79494fe6116921ff7b4d1287bd67451ba7498ecd))

### Refactor

- Rename SiLUWithTrainableScaleBeta to AdaptableSiLU ([fdf34b0](fdf34b0c04eae35af3867c8f9109cf24400b4d33))
- Add test for the fixed point silu ([2b0da94](2b0da947ff8cddf282294f4139a74b4b723cc4cb))
- Split batch normed conv1d and conv1d layers in seperate files and add parameter getters ([924e5f3](924e5f3646c151667c881360ed68aad890dc5a67))
- Split layer.py into multiple files to improve readability ([37128c9](37128c95d0a28268b1889783b31e536074d01ab9))
- Remove unused import ([24164b0](24164b0ff52f9e26a21a2342c9be2d5079a270e6))
- Remove not necessary fixed weights ([d9482ce](d9482cec23e20d4b370be8037fb528552b92c2cf))
- Rename Translatable protocol to DesignCreator ([e60bf7f](e60bf7f0688ba6e875d821c8cab4c34f4054cdec))
- Rename translatable module to design_creator ([0aa4d72](0aa4d72458aef4a91c87334048c357a276627d3d))
- Remove failing test ([e56a07f](e56a07feb90b8853f3b2901503bbe034fa7b4f16))

### Style

- Beautify commit # [4b05c4a](https://github.com/es-ude/elastic-ai.creator/commit/4b05c4a847ee33048b9552a93f816d6fec3c404f) ([c10028b](c10028bd5a2019c8de67f95bf1c25c930c5ec8d0))

## 0.57.1 - 2023-08-29

[6858178](68581787183a4ce4058d7db87e366c1376bdf08c)...[a5aecb3](a5aecb3cbc65e1231d1e2de957f28a1fdaba427c)

### Fix

- Try to exclude test files from build ([f282ac0](f282ac06aae45451d3787d74cda54b51e7f28200))

## 0.57.0 - 2023-08-29

[535c4c1](535c4c16f9c82922f4c6dc07c9737995f66f72fb)...[6858178](68581787183a4ce4058d7db87e366c1376bdf08c)

### Feat

- Global math operations depend on math operations of supported layers ([127ffdb](127ffdb29853587e4f819d75077e524e7a168bc5))
- Exclude tests from build ([72b8e0a](72b8e0af0e3fddc154be5763b26f5174cc49d7f4))

### Refactor

- Move unit tests to the elasticai package ([bb0ab8b](bb0ab8b8b8636c07318bae5e662836a07b5f33ec))
- Rename test files from test_*.py to *_test.py to improve readabilitly ([b7d3557](b7d3557338617c90b9b70459c4eeff12cc1c4623))

## 0.56.0 - 2023-08-28

[882c9c3](882c9c3bf5fc1173bd7722bfc2d9581e86575828)...[535c4c1](535c4c16f9c82922f4c6dc07c9737995f66f72fb)

### Chore

- Remove fixed commit scopes ([9f6b2e7](9f6b2e793e7a952f5ddff7efc89677a2f00c935e))

### Docs

- Start glossary ([b2d82cd](b2d82cdcc663d13492b611a700321c4bbcf452be))
- Update minimal example to reflect the most recent changes ([8f122a0](8f122a04699c42ff7abb7179d3cb7412cf94c0ef))
- Add glossary entry ([8855e02](8855e0248e97a2628c1ff73e69538c969f52d685))

### Feat

- Add layers to __init__ file in fixed_point package to improve usability ([93917d8](93917d8654794b7f5baa005e989f2984d6c846e3))
- Add a public quantize function to allow initial quantization of model inputs ([c8a170c](c8a170cb1ba529855790dcaf2dad97c38171174e))

### Fix

- Outdated imports ([650a71e](650a71e36f9c12db1160b86f82ad4d37715b19d7))
- Remove deprecated layers ([d55c041](d55c041e0141fe30bf3038074c061704ed057682))
- Fix broken imports and other errors that leads to failing tests ([6eac561](6eac56189e1e449d378867a4b4d8003967f48689))

### Refactor

- Remove mlframework/typing.py ([6858537](6858537f7906b75495376828c4713690b14cb461))
- Remove empty folders, start docs improvements ([28a9a2d](28a9a2d96521fbcb53ca889b746470bb99aef20f))
- Improve separation of core packages ([b5f469f](b5f469f1fbb354547560c9fefcd88e271382ae91))
- Restructure packages ([2fa7a4f](2fa7a4f58a481868edca1bdd3a568686130873dd))
- Move batchnormed layers to their base versions ([b1e0feb](b1e0feb6e45fd4f300065397ab2698382135c4b5))
- Separate interface for conv1d ([76ba0ac](76ba0ac054d7acd07ace2eb9875e9bd3473eeca3))
- Rename and move modules to fit our new scheme ([8effe1a](8effe1ac03fceebd324e4ad07f9d305c8e7d0c08))
- Remove arithmetics and autograd_functions from base_modules ([1e23c74](1e23c7406ffb0c0ab0a62aa31f8d61a502a4886f))
- Rename inputs parameter to x in forward parameter lists ([314b747](314b747e72509a2d34e72cc1d7738e2e26c18bd3))
- Adapt the structure of the tests directory to the latest changes ([b3cd5cc](b3cd5cc5c3cb68e7b5adb8136322426830d6db40))
- Reformat code ([1fb5ed6](1fb5ed6663424a98a36db3ad6ef899d62c74b75c))

### Style

- Beautify commit # [fcb153e](https://github.com/es-ude/elastic-ai.creator/commit/fcb153ea3aa32a73e07dd1f71d148634698a6cda) ([6515ab0](6515ab0225bd4b55b4e8a7ad1a5e4acb2d397ea3))

## 0.55.2 - 2023-08-12

[7642f80](7642f80fec4c7eb59ea9a3f6b403731bc1f304a4)...[882c9c3](882c9c3bf5fc1173bd7722bfc2d9581e86575828)

### Fix

- Add dummy batch dimension to meet the requirements of the batch norm ([0f499f0](0f499f048c0606de3e14163f16e8bf049708e6f1))

## 0.55.1 - 2023-08-11

[e1b7d90](e1b7d909bf7e496690a34a53a23276ccf4cc1d90)...[7642f80](7642f80fec4c7eb59ea9a3f6b403731bc1f304a4)

### Fix

- Fix non existing in_channels variable and remove unused import ([0e73c2d](0e73c2dc6876772d0caba46639af77bd5ac53b62))

## 0.55.0 - 2023-08-11

[b276d40](b276d40440bcf6f44b270d51bd44cccfe47fb1ab)...[e1b7d90](e1b7d909bf7e496690a34a53a23276ccf4cc1d90)

### Feat

- Implemented batch normed conv1d layer ([cd6836c](cd6836cc72b3fecee1b522f9b8934fabefd46d63))

### Fix

- Typing and errors ([af6f859](af6f85913fd6111bcc7164a106a9cbb8d4b7b9a0))

### Refactor

- Set alias for Design/FPLinear ([559395c](559395cde0dca73344cd162df04fea510a621b49))

## 0.54.0 - 2023-08-09

[85fe235](85fe235a2329c110bd3842757ef4ae14f7aa37c2)...[b276d40](b276d40440bcf6f44b270d51bd44cccfe47fb1ab)

### Fix

- Use same bit width for all rom values ([cd609e6](cd609e65f306e62110fbdc4113f4bb330f960f19))

### Refactor

- Rename precomputed monotonic increasing module ([ab8dfdf](ab8dfdf4c19646ae14dd787203a380eda47c281d))

## 0.53.0 - 2023-08-02

[2d8ecd7](2d8ecd7f02f2bcd1eb4b2e11572d5f54cf40c99e)...[85fe235](85fe235a2329c110bd3842757ef4ae14f7aa37c2)

### Feat

- Implement fixed point one dimensional convolution ([2ea9389](2ea9389a37eac7be62e26a9727b8824b47fc2085))

### Fix

- Fix missing parameter in tests for conv1d ([d8f8d4c](d8f8d4c40ec1576c5dc58a38b2b80d9d4130b4fd))

### Refactor

- Simplify string ([de8d3ec](de8d3ec98a6105d48630a0b2e6d82f15c3e75a9e))

## 0.52.0 - 2023-08-02

[4f690ee](4f690ee8b2209914bac8e0a7a175756b7832342f)...[2d8ecd7](2d8ecd7f02f2bcd1eb4b2e11572d5f54cf40c99e)

### Feat

- Implemeted base module for SiLU aka Swish activation function ([93b5954](93b59544c1d164de2a9f9362f0aefe1aaae8d7d8))
- Added nn module for Swish activation function ([b7579c9](b7579c9f1111521064a7fd0366647da7a45e2d7a))
- Added Swish activation function precomputed ([fd487b5](fd487b57bb7d3e7525f935f1533f815e58f1dc0d))

### Refactor

- Changed names of learnable parameters in the swish function ([bb2b7a8](bb2b7a81bf6365a54590575f39a447b6cd769cd9))
- Delete files that aren´t necessary ([53aebf3](53aebf3702c9c3511ef81e8fd9a1fcca018bf26d))
- Removed unnecessary file ([fe58c0f](fe58c0f22d38284e289542b6f3e58fbff60963f9))
- Deleted unnecessary File test_silu.py from branch ([80e8919](80e8919078ba32dd9af0146a94bd38b63bc761b1))

## 0.51.0 - 2023-07-28

[04bf4ba](04bf4ba8ae38c96008475c1c5ec22ffa63f94de1)...[4f690ee](4f690ee8b2209914bac8e0a7a175756b7832342f)

### Feat

- Rename custom float to float to improve readability ([794bfe7](794bfe79a6a821050b33cc246e9e1cad09e7e682))
- Add debug messages ([ee09864](ee09864686d87471617aae4ae65118096d31a6ff))
- Enable debug messages ([36ca597](36ca597ded38bb3c5343e872ed7cf9cb09065a6f))
- Seperate semantic release run into multiple steps ([475b425](475b425910c4a124c34ca9a68fd5c49b4789541b))
- Apply proposed semantic release migration procedure ([d5ea981](d5ea981cd8852e5790c77d9667187168c34c81e3))
- Increase semantic release version to v8.0.4 ([bb29612](bb2961243f20ade3e7c4a142601f58fca6e9b5ad))
- Revert changes and explicitly set semantic release version to v7 instead of v8 ([2ecf0db](2ecf0db3c22ce034c4f36a26c96027f0229a4bf0))

### Fix

- Remove not implemented jvp function ([0ea4834](0ea48341c02d116dd3ef2a94e0997ce8e0641b60))
- Try to fix semantic release ([0eab187](0eab187389b3d435be473671d4a593ead8586e78))

### Refactor

- Remove noise comment ([f6be240](f6be240b03484876627f5f7de5198fd1332d6ba7))
- Remove newline ([33fa0a9](33fa0a932b5c2126a004b429702bcda72e696069))

## 0.50.0 - 2023-07-11

[aad3340](aad33400ae4879477375ecb4bf507033eebca4b6)...[04bf4ba](04bf4ba8ae38c96008475c1c5ec22ffa63f94de1)

### Feat

- Implement RoundToCustomFloat autograd function ([0794a8e](0794a8e900d6f87edc03dbd71162e7300e13b5ae))
- Implement custom float arithmetics ([b72713e](b72713e3db1e15957e865ed95216a2f180523114))

### Fix

- Return wrong number of values in the backward pass ([6bcfa4e](6bcfa4eff8d9b7c0c0461a61800ef68ef6b0cb62))

### Refactor

- Rename FloatArithmetics to TorchArithmetics ([5cd7a3b](5cd7a3b6913b14456f524a9486bac2c42dc72412))
- Rename CustomFloatArithmetics to FloatArithmetics ([824b029](824b029971789c951a243937d942d7597225e829))

## 0.49.0 - 2023-07-01

[1211442](1211442c4b3de0dd47a2491530fd70f14a02fe3d)...[aad3340](aad33400ae4879477375ecb4bf507033eebca4b6)

### Docs

- Complete table of contents ([cf0ef63](cf0ef63eb628521f14406fb7d59cee53c71c8d60))
- Add minimal example that demonstrates the usage of the creator ([64030f2](64030f2eb129ff8275022ab0b8bf4945d42626a8))

### Feat

- Update readme and add small improvements ([8f2bbd0](8f2bbd093e18c15421abab20ecb0f9afbc6d12a1))

## 0.48.1 - 2023-06-24

[f9a851e](f9a851eea1a3c6b458b2432ba83391aa3fc3ed48)...[1211442](1211442c4b3de0dd47a2491530fd70f14a02fe3d)

### Fix

- Only create coverage reports in PR ([1bd728f](1bd728f4e8edb6595a35dafd71c5d68263a7358f))

## 0.48.0 - 2023-06-24

[da12e09](da12e09be6b0198577720191340266c94244cf39)...[f9a851e](f9a851eea1a3c6b458b2432ba83391aa3fc3ed48)

### Feat

- Use binary values instead of hex values to fill the rom template ([af56c02](af56c02da42433c2db1a9a2a6ddb3705d213d765))
- Add pytest-cov dependency ([a737729](a7377290ffee7359f6f8c0392960d7038fe2a73b))
- Add coverage workflow to create reports ([3f6caca](3f6caca6a626923ec3d8078320fa9b70092495ee))
- Only trigger coverage report when pushing to main ([b4b23c9](b4b23c988803165895c14a8357427a3069f09233))

### Fix

- Use poetry run to run pytest ([7058e42](7058e42cc7fa0849841578f2bafd6a3fc6155f2a))

### Refactor

- Improve readability ([e4de568](e4de5682419829675a92ff95f8e853dc28cf181e))
- Remove unused to_vhdl_hex_string function ([24ccbf1](24ccbf1a1d9ff3d270faba19581a6f72eadb751e))

## 0.47.2 - 2023-06-23

[cc4279c](cc4279cfdff992551fb274818964c74e216c64ec)...[da12e09](da12e09be6b0198577720191340266c94244cf39)

### Fix

- Fix error when passing a cuda tensor to the IdentityStepFunction ([7f49617](7f496171a547bae17c69976c35d437428022447f))

## 0.47.1 - 2023-06-16

[f0e1eee](f0e1eeefa6f972574a9324885f00af052924f295)...[cc4279c](cc4279cfdff992551fb274818964c74e216c64ec)

### Chore

- Add do_not_commit path to prevent files from being committed by mistake ([af13e16](af13e1687f57fc3545d0c114263ed439b78973cd))

### Fix

- Remove wrongly committed files ([4fdea0c](4fdea0c9ff2db5e8af3f208bbd83d995332d5b85))

### Refactor

- Merge fp quant and fp dequant into a roundtofixedpoint autograd function ([b986a62](b986a62ea7a0a58e6479aa5082ddd2de11ed27d7))

## 0.47.0 - 2023-06-16

[f029667](f029667c2c9f61acacb0638f1a9bc85177c9e553)...[f0e1eee](f0e1eeefa6f972574a9324885f00af052924f295)

### Feat

- Simplify project structure ([81cbcb3](81cbcb343b26473290609c7715051059127a924b))

### Refactor

- Remove unused manifest module ([55f8e6d](55f8e6deac74d953b97b031a22e0dd9a73ecf20c))

### Style

- Beautify commit # [7beebdb](https://github.com/es-ude/elastic-ai.creator/commit/7beebdbc67074dc6f8e8a0320563385ee49a7915) ([1c7fead](1c7feadd0825be6648702c7ecffcdb1c2ce974f5))

## 0.46.1 - 2023-06-13

[dd9a1b9](dd9a1b95a9a7a1a3b5f0074c18ed9c3d8aec3e73)...[f029667](f029667c2c9f61acacb0638f1a9bc85177c9e553)

### Fix

- Fix wrong port definitions ([9a4c8af](9a4c8af6f8f8be2bf6fff49c25fc0ca12cbea45a))

## 0.46.0 - 2023-06-13

[8e6c6fb](8e6c6fbbae0e5b3d32eecc95a0877a5c288218fe)...[dd9a1b9](dd9a1b95a9a7a1a3b5f0074c18ed9c3d8aec3e73)

### Feat

- Add the ability to sum over dimension ([c45c0e6](c45c0e676e1df70bf99c4c943874168781ef2a93))
- Test that conv1d uses different arithmetics ([7eb01db](7eb01dbaa2afbbb02410e6fc6272ba02fec7878a))
- Add conv1d function to arithmetics ([1cab190](1cab1901e324eb100f1cbccf6d54fae429210b33))
- Use conv1d arithmetics function to implement conv1d module ([69778be](69778be7fd1becab2ad5099ebb8d64d4a0db0de5))

### Fix

- Fix some syntax errors ([3997bbd](3997bbdb134a94defd4e32ad1a2eb3aa236d6b96))
- Quantize weights before inference ([61153e6](61153e60d6854bacf0bd2501d96efc3f6e62714e))

### Refactor

- Remove debug print call ([f85172e](f85172ebddfceb98a7c661cd3f57db60b19b61c0))
- Improve readablility ([004c736](004c736cab22b4e8eed5eb867c203b4b62e7e235))
- Remove redundant tests ([c828d53](c828d536110205b3e00f61a33e31d0cae1eaee6f))

## 0.45.0 - 2023-06-10

[6839439](6839439f318cd10e304a625ba3120c81dc292be7)...[8e6c6fb](8e6c6fbbae0e5b3d32eecc95a0877a5c288218fe)

### Chore

- Update dependencies ([a2558b5](a2558b5649d5416c730cfdfebdd4d38ce48a6a88))

### Feat

- Simplify usage for the elasticai.creator.nn.vhdl package by adding layers to __init__ ([2c7c968](2c7c96858ec9d935389a960baee46e8c506f9b5c))

### Fix

- Fix broken import in base template generator and move it with its template to own folder ([9eb1f70](9eb1f70cff10e075712d5bf7e3fc9fcfed2aae19))

### Refactor

- Remove unused template resources ([d58f267](d58f267772839df6c254b9d749b8e5653b9a20e1))
- Rename sequential layer module according to our convention ([ae1da5e](ae1da5e5aced255e38f0c13691a1d42f90dd5cb3))
- Remove unused and redundant port definition ([b376b75](b376b757f6dd0e6400813688a2dfdf6ca392a6f9))
- Rename template and remove some newlines ([707310b](707310b3202ec1b48f847a228455f8cd77436219))
- Remove some newlines, use create_port function and fix wrong template ([1bc4a70](1bc4a70f173c9f380a76438842ba6708d1659aad))
- Transform bdd test to pytest test ([475ec7b](475ec7bd12ed0f43b65438a2ef62aa97d3ca8b14))
- Remove unused pytest-bdd dependency ([e9203a0](e9203a0223ef3adfcbd40af841e569438684e1c8))
- Rename monotonously increasing scalar function ([baff8b2](baff8b2fd8569c60b51906458c2d541e1371f111))
- Better separation of designs and modules ([44f22ae](44f22ae25a02c0c4810e64c970cdc5dd28135c89))
- Create rom design folder ([9e40f5f](9e40f5fa9b40c2542e4ef99cf02d1b6004ad2a60))
- Remove deprecated documentation ([349a9f8](349a9f866e001ce0494f9876d894ef0c5833817d))
- Remove unused base signal definition ([14dc275](14dc275beea7f3c757433eb9b3872c895fc6fca3))
- Rename ports module to port_definitions ([b5a64b8](b5a64b812145a34dd1dd0d20cb2ca31f18804a1f))
- Use container types from collections.abc instead of typing because they are deprecated ([7a45e67](7a45e672cdcc47b426a57a8297febc8aa9664744))
- Remove unused imports ([e8881d3](e8881d31e11d3e27489322deabb3c29d420e568b))
- Use Identity class from base_modules instead of torch ([8f179f0](8f179f0bbbb2510b294665e0502715b6b69346c8))

## 0.44.0 - 2023-06-09

[d317c39](d317c390c7394785626886970de8f6dd3b37f470)...[6839439](6839439f318cd10e304a625ba3120c81dc292be7)

### Fix

- Port def and impl of monotonous function design ([2d423d4](2d423d46faa86fbf43cb8ba1d01aafe92c5bfa23))
- Use new Sequential constructor ([6bb111b](6bb111b748567502c23a48a52d7e477645969996))

### Refactor

- Cleanup imports ([c402a03](c402a031f5996c6f7a1b3a5199e1cf9697e7dc5a))

### Style

- Beautify commit # [95ca255](https://github.com/es-ude/elastic-ai.creator/commit/95ca25571e9757d932a45749e9cf92531c13ab36) ([cdf44ce](cdf44cec1a9a656ce6b3a9d19a717a9e7163d1b6))

## 0.43.0 - 2023-06-09

[d37ed1c](d37ed1ca6c6e0a49c7c25ca11e06f9a9566fb70a)...[d317c39](d317c390c7394785626886970de8f6dd3b37f470)

### Feat

- Introduce FPMonotonouslyIncreasingModule to easily add new activations ([b78c922](b78c9225f7f70ec329bee5705c11d9e7b1392c41))
- Add tests for the FPMonotonouslyIncreasingModule ([9ba64ae](9ba64ae3d253db76a6368c5e561ce28bcec2aab5))

### Fix

- Increase default sampling intervall ([07620d3](07620d3e2ee9db1bc6aa081a15274cb79b5ee4b0))
- Use elsif in lookup table ([f375ba3](f375ba3784bf92887e689f77f592dfc2fa2c7e2c))
- Set correct signal names for x and y address ([5354a2a](5354a2a0e85bc0788f5d74377c1a685e9d0e0de7))

### Refactor

- Move all arithmetics to arithmetics folder in base_modules ([de0fd46](de0fd460eae7d7d155188d2e73dd4cc82b913718))
- Remove unnecessary tests ([c0756b3](c0756b3d7a7468aa0e3d7c55e126170790bae076))

## 0.42.0 - 2023-06-08

[59e48f1](59e48f1e0b609088fc7aa43479cfa5f7a785c9a8)...[d37ed1c](d37ed1ca6c6e0a49c7c25ca11e06f9a9566fb70a)

### Feat

- Make sure that inplace parameter is fixed defined ([79b7a1e](79b7a1eea0cb71f5a838cfebf02970927410f594))
- Add working hardsigmoid implementation ([db03ff0](db03ff080f878c9b9fe54303ead97c673022f3a1))
- Reimplement hard tanh activation function ([9b86f9d](9b86f9d440cc991d624a6f3492a3caf7419bdbf3))

## 0.41.0 - 2023-06-08

[1800a76](1800a76e78e9289eeb9741da75a367744c0fa4bb)...[59e48f1](59e48f1e0b609088fc7aa43479cfa5f7a785c9a8)

### Feat

- Add fixed point ReLU module ([62c1555](62c15557fc515c89644c674aef9fc39d22ab672f))

## 0.40.0 - 2023-06-04

[7b9b42d](7b9b42dc6eb1414666dcdc32b954a8a9acd6bc02)...[1800a76](1800a76e78e9289eeb9741da75a367744c0fa4bb)

### Feat

- Add a function to easily compare tensors with pytest ([24e737e](24e737eaea48044df3e8addaca0d1cc804a3b6f4))
- Implement autograd fn to map inputs to a subset of inputs ([26c6ec7](26c6ec7a203eea4fed4c3eb3d5c3e4893acb545f))
- Rename autograd function and pass step lut to autograd function ([d607e98](d607e98bd14dfa1ae23e9726b2046baaede21361))
- Pass step lut to identity step function and improve readablility ([c1b6747](c1b67473c33ddc27590068472dcff6969f9e7135))
- Implement bufferless component interface for precomputed scalar function ([f701a57](f701a57db54e0d5f3e5e43047725b28646cb5f15))
- Add quantized tanh implementation with lookup tables ([3a1fb10](3a1fb10944e566ca33e3e745b939b6700421fdb9))
- Improve performance of the identity step autograd function ([46f036c](46f036c8fb2d007d21e32214ac92d4d9aa2fe9d1))
- Simplify the use of the sequential layer (same as in torch) ([9fad15d](9fad15d774f3573fb26f168295f9bd2ae5cdd046))

### Fix

- Fix missing creation of a subpath in the save_to function ([2a4dbdf](2a4dbdf2f6fce4de567281002dd4640ff3ae54ed))
- Fix that last io pair was dropped when calling save_to function ([2bc46ac](2bc46ac9c535b65ef7a3dc5cbe12b27d253c3b37))

### Refactor

- Move torch dependency to base_moduels ([06d1aca](06d1aca6e3ca95a1e371253aa97dee831119250c))
- Remove unused base modules ([97d1e7d](97d1e7dbc181fc03562ccbcde976eb9e661c381e))
- Small change of the folder structure ([58783a8](58783a83a891d85c50c43a6af2ac3efa3e634657))
- Remove unnecessary tests ([23f78db](23f78db7aec7efeef669a32ebe76ea3ebcb6b133))
- Remove default sampling intervall ([9d7caea](9d7caeae98408d2eaf0c97032dae0b5b4b312429))
- Remove unused import ([4de2055](4de205551938c7a284af78b5c2c418fdf95358f6))
- Change indentations ([d5f5bf0](d5f5bf07b85d7b1902d474975da58d29bc615f6d))
- Remove the leading underscore of the class name ([6643bf1](6643bf13dfbe50f7b98c0a49a238041c49fa8b89))

## 0.39.0 - 2023-05-19

[f06c144](f06c1440f8e45535b9d1a56142356a9acea31bfd)...[7b9b42d](7b9b42dc6eb1414666dcdc32b954a8a9acd6bc02)

### Feat

- Make precomputed scalar functions bufferless ([89986fa](89986fad041c89d0543fe9a22946e5f5f49e2b61))
- Port expansion/template based on autowiring protocol ([0d14618](0d146181c8b789b09871af43654ca2d83ea55ddb))
- Add basic vhdl parsing ([5df2a3f](5df2a3ff4e9ba7ec33398a267cd983ad886d1fe7))
- Add standalone parser module ([5a9b141](5a9b141285fefecf61f581417061428cda382ad5))
- Support parsing partial files ([8170012](817001208b774e57cfb27fb4d4ee9d704541c9f8))
- Add intermediate symbols to rule definitions ([624b310](624b310fc9beb130902fdf3269e3f30714fe0c3f))
- Add AutoWirer ([f4159c8](f4159c800fe54cc0fe73fbebdf2ac0410ddac635))
- Check for autowiring protocol violation ([3f17e00](3f17e002e050dc92516e4ff5468041f06ebd6760))
- Add experimental precomputed tanh in fixed point ([0e76d03](0e76d03b6d0f23d8932b94bb7728cbeea2de0289))
- Implement batch normed linear layer ([9322f6f](9322f6f699f9884273c3f9815b9a026c9f7840ae))

### Fix

- Correct tuple type annotation ([f0e7da0](f0e7da0cf186015004970102f2b9b57a9f839585))
- Adjust tests to follow previous change ([c328bd5](c328bd565d6ba84a9d1fab788051c3e884ea2094))
- Remove obsolete parsing functionality ([7f85d05](7f85d05aa3da2e0fd7c266bfc9c1aad573adecc4))
- Children of sequential layer determine signal widths ([3dd5c0c](3dd5c0cc4f7a52c7b3a86cec437005b86aa0a509))
- Remove dequantize ([c111022](c111022854ce6965b705b3a3de296e032d7ff107))
- Allow to set affine and bias equals false in translate function ([b351284](b351284335a77caec838a8f4ea57684e429cc35b))

### Refactor

- Remove obsolete module ([5adc999](5adc999c3f4fb5a45e569680fa466694127688da))
- Make identity layer/design names more specific ([0aed47e](0aed47ebd3dbd784156a949822b8fc7c117e07c0))
- Remove obsolete test helper code ([17e4e12](17e4e1250c1b94b3f72ac9dba57f7ee66825f381))
- Pull up tokenize functions ([ace6f1e](ace6f1eb5d0162d7454d56a5baf6f3fb59f3dc06))
- Pull up parse function ([1b8f187](1b8f1874eff63130e71c1754257d5bb3d05bb827))
- Move sequential layer to nn.vhdl ([caea325](caea325588f8c87cc28d5df248129b0e73111e3d))
- Move binarize autograd function to autograd_functions folder ([03d5bc8](03d5bc86462b36be30c2887593360ec48a908ab1))
- Rename FPLinear1d design to FPLinear ([238f167](238f1671a28b9b5735ca7e01360d4dda7122a2a7))
- Remove redundant quantize function ([02094cf](02094cf412f2846821c9c2925bedcdc585fe8a8d))

## 0.38.0 - 2023-05-09

[b752226](b752226a16f65806bad09e88a2f65f7fffe43168)...[f06c144](f06c1440f8e45535b9d1a56142356a9acea31bfd)

### Chore

- Remove unused workflow ([dd08e08](dd08e08b0af74c4d7ba927c892de6081717657db))

### Feat

- Write function of InMemoryFile and OnDiskFile now takes Template object ([a867ea1](a867ea15980b8ca1390327f2999c4d7b91ef3041))
- Add function to get all unfilled variables of a template ([d635cb6](d635cb6098735b451aea259a8a6f15619bfcd64f))
- Add check that all variables are filled when saving a template ([c988d2b](c988d2bc203790ba8ab900e8a2de6996b22d6fcb))

### Fix

- Fix not inserted process name ([dbabea0](dbabea07c888a5309d9ca55cd2c01ae0debea57d))
- Add variable ([229d452](229d452d0c2f798ee1dd0124f50be8f01d69ede4))
- Remove broken lstm implementation ([c524ca2](c524ca20cc49333007c4e0bbfa167912580e5c01))

### Refactor

- Temporarily rename template class ([6fb83a2](6fb83a2d773bb474bf96f4c248de8537f91673aa))
- Rename TemplateConfig protocol to Template ([33d01ee](33d01eef31e7c9cb919a9684150dfba8ce1c60a5))
- Remove InProjectVHDLTemplate and InMemoryVHDLTemplate ([e625399](e6253997447b0976de4ed60ec671de80ec6740a6))
- Remove RawTemplate class ([eb91cd8](eb91cd81475a6a9aa94fc8ab4ccf3457cef55d01))
- Remove deprecated and broken relu and tanh implementations ([286686c](286686cd6a2a185a94c03585f41d15dea794b1a2))

## 0.37.2 - 2023-05-07

[4797aaa](4797aaa00b066104eb17dc8977bd6f47ee112396)...[b752226](b752226a16f65806bad09e88a2f65f7fffe43168)

### Fix

- Try manual publishing ([c8b6c35](c8b6c355896c1f3b0630c227af8414f281b5d3ff))

## 0.37.1 - 2023-05-07

[7fe6bba](7fe6bbaf37ee509a703ed4eeb446206eb1e3024c)...[4797aaa](4797aaa00b066104eb17dc8977bd6f47ee112396)

### Fix

- Try to fix semantic release ([2625e89](2625e8982c021cbf5b778e95194cc53170ab0afb))

## 0.37.0 - 2023-05-05

[f2707c4](f2707c47f99a1f9d11addbb3fa966054ed1a0b8f)...[7fe6bba](7fe6bbaf37ee509a703ed4eeb446206eb1e3024c)

### Chore

- Add  force-publish workflow ([b59268d](b59268d15b8ef605c6dbb48e606f5b1ad746548f))
- Update force publish workflow ([9a0a7ac](9a0a7aca438f92e728c0310ec16adb0ded902f29))
- Update force-publish workflow ([c7b011c](c7b011cd289baa1615cde11224f2a0ec25221e15))

### Feat

- Assert that all inserted variables exists in template and remove AbstractBaseTemplate ([51f1a08](51f1a0883a8d0a54caee66080ef85f84049ad806))

### Refactor

- Remove unused parameter ([89ca654](89ca65467a983230a1dc54d8b1502e82185f2acc))
- Remove duplicated test ([cfd304e](cfd304e630ba4f13ee87fc074c7d05fd99b1c98a))

## 0.36.0 - 2023-04-26

[2c91d42](2c91d4293eee0aab1af6b2c796936db8e0d93807)...[f2707c4](f2707c47f99a1f9d11addbb3fa966054ed1a0b8f)

### Chore

- Adjust main.yml ([93550cc](93550cccd7eda401dc7f759da8efe048661c2573))

### Feat

- Introduce abstract Translatable class ([5d9fa2d](5d9fa2d167a8c46c301bb4a0da25718b1fcf0dee))
- Sequential layer can have a name ([9e46938](9e46938e9e5fc6960e70bef26aa72ec51566a007))
- Test all subdesigns generated by sequential layer gets a unique name ([009405b](009405bc64cd5e8a86909330bb450ee58ee98289))
- Test signal definitions, layer connections and instantiations separately ([65201c8](65201c83bae07c62efcd705f67f34d9ff88da557))
- Add tests for sequential model with two layer ([df73a4f](df73a4fb27a8867a4b633c4ffdd737ead34d2f16))
- Autogenerate sequential signal connections ([6dfca07](6dfca078b735a3387b65c20de601426ea27371c6))

### Fix

- Add missing save_to function ([ef24ee2](ef24ee21672099359867bc4a74f5804af0c10158))
- Fix syntax errors ([f9b57e4](f9b57e4f8173dc0bd52c21b1da351304ceb5a122))
- Fix syntax error ([396f5c4](396f5c45b382454d6cc97e4be573fcfe45a4592a))
- Fix that test ignores parameter ([f448919](f448919ff4882696c0991d6aec3608616e258596))
- Correct expected connections ([2fb0f8e](2fb0f8edc45a7a38e2a9b7433dee90f139b10006))

### Refactor

- Remove unused translatable protocol and rename module ([9d59f8c](9d59f8cd533b32baf6f90365e0db5a8b18d1c5a7))
- Remove unused import ([602c137](602c1376cefe7dc4a95ef7cf04b9f67b0e2cf1e3))
- Fix/add missing type annotations ([d47a8c1](d47a8c1c8919066e557a702f3bccc3928f35fa69))
- Use identity instead of linear layer to simplify test ([28a75c3](28a75c337734b6bed887b1a3f9fc0369d92d330b))
- Rename FPLinear1d to FPLinear ([5550dd9](5550dd97956171f53edc59e534dd02161c463133))
- Reduce code duplication ([ae65808](ae65808bc66ebd2982a80ec3b6c5d70f749723d8))

## 0.35.0 - 2023-04-17

[1dd7d74](1dd7d74f0d4795ca6141da7b9b0957298b2a604d)...[2c91d42](2c91d4293eee0aab1af6b2c796936db8e0d93807)

### Chore

- Move to python3.11 ([389e4ec](389e4ec6d60dbf594026993bf8f7d94d4bea1da8))
- Upgrade to python3.11 ([f39c779](f39c7798f4ccc3799c707c8dcefbd176f9b6813b))

### Feat

- Add translate_to_vhdl function ([ba0edc2](ba0edc25b93075cbb2d104c2216dcc15df36c13c))
- Implement translatable identity module ([54327fa](54327fa3e45ca3617d642134ca8d842e7d2afc4c))
- Add indentations to template ([aa254d1](aa254d12f38712e798db9b31a5a58e197a44121a))
- Generate first base template ([a65d72e](a65d72ea1ad2dd87a0443b56711d11ce321d14b6))
- Generate template from manifest.toml ([51276a0](51276a01de5ff37bedc598f5c758e3dc681aa49c))
- Use fixed base template ([432dfd9](432dfd9518a0a33a7ba08cf95436f9472b274b52))

### Fix

- Set correct resource options in rom and fix signal definitions ([2c2964c](2c2964ceaa746163ebbeaef09181e09c06ecb4f2))
- Fix tests and remove hard sigmoid test in sequential test case ([a1ada6f](a1ada6f0ceec750bb80abf866d28f96719f2f1f9))

### Refactor

- Remove unused imports ([d9592ec](d9592ecb3677ba8050cb737bbc112987e72f25b5))
- Remove superfluous module protocols ([4e25dc6](4e25dc65dfa0c226c298f5e589a6c887d72a3c19))

## 0.34.0 - 2023-04-06

[165f434](165f4343832b770386b1c98649aa69878bbaaf33)...[1dd7d74](1dd7d74f0d4795ca6141da7b9b0957298b2a604d)

### Chore

- Remove unneeded import ([e3df52a](e3df52a091e4673460f7b1ad733d766bad4afd02))
- Add mypy and pylint to pyproject.toml ([aad5549](aad5549c7bbfbaf648fc3bbab0f77cd6c0ad49ca))

### Feat

- Binary_arithmetics ([54e38d5](54e38d57f27db2d8d0baff5fee3c35a91e26ecd9))
- Make precomputed scalar functions use unified interface ([6b59da5](6b59da53a896db7676119de2f74129bcc47287ed))

### Fix

- Correct import paths ([169f868](169f8686108845702f01482170df53e3fabbfe8b))

## 0.33.3 - 2023-04-06

[463ff20](463ff20af4c7f5aa97c1cb9453713804e6ddef2b)...[165f434](165f4343832b770386b1c98649aa69878bbaaf33)

### Docs

- Remove deprecated documentation ([11b9945](11b9945bf3b6bf96899a09751963a93eb98d846d))

### Fix

- Set correct rom names ([9570826](95708269900ca99b79da9ba37078f593724e5d17))
- Remove DualPort2ClockRam design ([f9224c6](f9224c6809b3a6f72bfe0405419de494b099b17c))

### Refactor

- Rename nn to base_modules ([44207a8](44207a8f72e426fcd1cb4acc5b3c53c4ac8fa2f2))
- Rename translatable_modules to nn ([333ac57](333ac5776788367ed3a8c17632fa20e11556f43e))
- Move hardware specific lstm parts to nn package ([bfe575c](bfe575c50291388eb2f8b243d3411ff9e847490c))
- Reorder class definitions to avoid the usage of quotes ([780c1fe](780c1fe67d18893400226e8acc6e77504da6a6ad))
- Move lstm designs in designs directory ([36a807b](36a807b00794bac42a5018759e2ec09238bf043e))

## 0.33.2 - 2023-03-23

[aea7083](aea7083d41dd77f30849efd56aa5f493614a4d13)...[463ff20](463ff20af4c7f5aa97c1cb9453713804e6ddef2b)

### Chore

- Allow all torch versions >= 1.11 and < 2.0 ([7321d7c](7321d7cf5694588a607975d13958edbfa5a3b331))

### Fix

- Fix failing unittests that are using the linear1d layer and design ([ff582e1](ff582e185ea01cc6282cb4553e14701e88a9d8f8))
- Fix type annotation ([8da1107](8da1107b2640d695816c71dd3980c0783b522122))
- Add missing ROMs and set correct names in fp_linear1d template ([ad4c6f0](ad4c6f095102965ff1dffa83dab4f2cb9749ce49))
- Add missing rom files and calculate correct twos complement ([f700409](f70040956b7637844a471a5eff171d9cc6ba4c72))
- Small import fix ([07d2e29](07d2e29c36e60d35066d2145782223aa42d64519))

### Refactor

- Small file and folder renames ([9602a86](9602a868e6067889e2386c764e173c36f33e304c))

## 0.33.1 - 2023-03-15

[1fb0bcb](1fb0bcb25d068e5819e7be00f94062328aa444d8)...[aea7083](aea7083d41dd77f30849efd56aa5f493614a4d13)

### Fix

- Wrong fixed point config object used for linear layers ([3626113](36261136add4b4d378598dc8c9e858240f6557c5))
- Usage of lstm output in lstm_network impl ([2e16141](2e1614184cdaa073fdcc686b891748861fe5c7cc))

## 0.33.0 - 2023-03-15

[a640437](a6404377d7f47c699be41fcd5ce3ea2f1f1db43c)...[1fb0bcb](1fb0bcb25d068e5819e7be00f94062328aa444d8)

### Feat

- Add rom design for saving weights ([75862b7](75862b7db4e64173daf7e6cdcb8413b0f510d396))

### Fix

- Correctly pad rom memory ([fe768d5](fe768d5f93c34ade65c24479c70f3528c66b0408))

### Refactor

- Rom design ([975ad7e](975ad7e139a15466338cff72cfedeedf0c532f75))
- Use rom design in implementation ([a8bfe4a](a8bfe4a2395a9bd81aa33f1989154f84a21bf001))
- Move conversions to twos complement from designs to translatable modules ([50ada18](50ada185de5a081295515e16773b7fefdaa107eb))

## 0.32.1 - 2023-03-14

[01be016](01be016011cef672e8489252f59e73dd36b533d1)...[a640437](a6404377d7f47c699be41fcd5ce3ea2f1f1db43c)

### Fix

- Set library for lstm_cell ([2b3a565](2b3a565039672ca89a1c5f593db5a5f32742f771))
- Typo in test for lstm cell designs ([2ffeaec](2ffeaecf3ba7c3c0946c57ab3bee92af55746887))

## 0.32.0 - 2023-03-14

[b605967](b6059676e662e63937af9eaf96feb7cb6c111533)...[01be016](01be016011cef672e8489252f59e73dd36b533d1)

### Chore

- Update gh workflow to match new tests location ([58b7151](58b71513d05aa0bbf34533dc72b070ceaee34e83))
- Update gh-workflow ([b1d714d](b1d714d4d408917ddd389db7fa29eed6c0230684))

### Feat

- Sequential layer with bufferless layers ([d7cea69](d7cea69ad0696f63e00762991e7407ad09d8a94c))
- Add support for single buffered module to sequential ([5402782](5402782c0c37a6838b77b19d8040d256217d72ba))
- Add linear layer to lstm network ([48982f0](48982f0aca675098b77edb2c8419b09ebc388835))

### Fix

- Tests and remove type annotations leading to deps ([75ed6cc](75ed6cc4f3a92b80656433b8209c0c932595900e))
- Correct values for x/y_address_width ([c7af1af](c7af1af71ef9319ed2ee7fffd7afcbaa5ffda580))

### Refactor

- Move modules ([24e522f](24e522fb10224bbd4065d841b2df97fa0f561021))
- Replace fixed point factory by fixed point config ([b5a08ac](b5a08acc11453ad550e2457836f1f4a2f5cbbae1))
- Start moving relevant tests to top-level tests dir ([577f43d](577f43d16a30fb1e6cc73c7dca7a4d6391559f79))
- Tweak module hierarchy ([40bc371](40bc371d6602c504ed6e69542ef3a51d525fda70))
- Remove code generation dependency on fixed point data types ([4d83d1b](4d83d1bc8f1a91de6dfd8995373155151d74fc25))
- Refactor autowiring for sequential network module ([431862f](431862f21b6f074021973a88789a654461ae269e))
- Lstm roms ([a2e08ec](a2e08ec2f1492cd0efc9f4e60b76b4a42c0d093f))

## 0.31.0 - 2023-02-22

[2e64ede](2e64eded06119897f311dff39761dbd7acd14d43)...[b605967](b6059676e662e63937af9eaf96feb7cb6c111533)

### Chore

- Introduce private package import lint rule ([b497e1c](b497e1ca3c512d2414cc0736305e19a867251741))
- Tweak import contract ([306de20](306de20163ad6e751b5e8d5e66601e90d1856b50))
- Update deps ([00700fe](00700fe92b86442cc7e0db29794fa78d20ba48f9))
- Add class diagram for vhdldesign ([01c63e0](01c63e02759ca71c93dc3f985d416d3ffa2c31af))
- Clean up external deps ([d1be65a](d1be65aee7144be24c79a280c93537115acd2e31))

### Feat

- Add connectable base in/out signals ([7ad67f9](7ad67f916815b692daddae98d4c93b9a5eb21641))
- Add logic and logic vector signals ([1947baa](1947baac032e1b3958344779a00b84615b5581a1))
- Introduce vhdl_design class ([20566f6](20566f600383ccb68fed60483bede9db5436913f))
- Add data flow node, sink node and source node ([9a511de](9a511de4d2618c3131abcd3c481b918ffa96545e))
- Add missing suffixes ([cb05d0f](cb05d0f3f8665ac98c0cff70cbb2dbd8d2a5b2f2))

### Fix

- Fix incorrect vector signal initialization ([3c68255](3c68255057dad325ab4ba89601f6f1e2384f0d95))
- Type annotations for tracing module ([da598a9](da598a92fc8f76b3c19d0b960d77122b82d171ac))
- Fix unit tests after major rebase ([3b596e9](3b596e9c20e302bbf42efda7577e01498c05bc6c))
- Typing ([b0bfa39](b0bfa39b98555b37f0d2626a235ac74987e2c9ad))

### Refactor

- Merge utilities for testing code ([333c09a](333c09a9b396f450e24d7d2390daa8b502b5cdac))
- Move file reading to CodeTestCase ([3cc9c5e](3cc9c5e4c67fea3e8bea566eeb1a30feea7c1b56))
- Remove unintended print statement ([b43befd](b43befdb529389a8cc8c08d087631ca45163f51c))
- Move code test utility files ([d390af1](d390af12f9658952fd08b4493b467ee820c45f5f))
- Rename test_logic_signals ([f817425](f817425f96895cdf52ff184f7cc32473e3c85fe9))
- Simplify architecture ([1f5f1f1](1f5f1f19510f6dd9282e5bdda5beab904b2328b3))
- Remove obsolete graph package ([ac53d76](ac53d7684135e3bab4d940d1c80951b297d19d77))
- Remove obsolete vhdl_design module ([d4e61bd](d4e61bd7440d42a878f7539af7c256d637c2b7ba))
- Simplify signals and move classes ([aacb702](aacb7021bcb83cb96053092640a7b7cdc6e2077d))
- Use relative imports inside packages ([ef8d588](ef8d58878058b2eb6ef5f177171350c6759132f7))
- Simplify data flow node ([82c8ba8](82c8ba825bfa3b5d367bc3d6f473d2055ef217d6))
- Remove/move/merge protocols ([8391a1c](8391a1c7e459bbf176840976a741317a28f3abd6))
- Only return file object from package without opening it ([2c57287](2c572879a98a4af72978bbd471704395606b96fc))
- Separate template from file ([73f00e0](73f00e0e2e1e6302f2d8325fe9075d9bd51c25a3))
- Remove deprecated vhdl.language module ([e29f6da](e29f6da7e76018dce7d32f9698a7973de6e5e832))
- Move modules/classes to fix dependency issues ([22564d7](22564d7ce4b05770d49078c0d5ce13fe3ace231d))
- Move more modules/classes to fix dependency issues ([ae82c14](ae82c143100ddb9a49a7cfae36d8ea5289789fa4))
- Adjust architecture in design.md and move modules accordingly ([236e6c3](236e6c3457cbbb413b8fd79015bfe1e97c49563d))
- Simplify signals ([884ad64](884ad648fde4381a4dd892542bf576a7cd2d090b))
- Simplify ports ([4bdf84a](4bdf84a4f72f1b99d89afa84de234c74a637fcd0))
- Remove superfluous protocol ([741c53b](741c53baf3ca0ee9ccb27d5cf5a64d172eac7781))

## 0.30.4 - 2023-02-16

[fdb1cea](fdb1cea8779ce59a81669b5cbe8e65e746214353)...[2e64ede](2e64eded06119897f311dff39761dbd7acd14d43)

### Fix

- Get rid of the duplicated suffix on rom component ([9cd0e0b](9cd0e0be9481a286820eea5c8d5bdc9d28fcc0d8))

## 0.30.3 - 2023-02-16

[f6e8c11](f6e8c119ff366f3269bddc733f9c9f4d82167693)...[fdb1cea](fdb1cea8779ce59a81669b5cbe8e65e746214353)

### Fix

- Linear layer template ([96bdf03](96bdf030ca4c27d67a4978e3b8609ef57c40a01e))
- Add rounding to prevent tests from failing due to floating point loss ([b7314b7](b7314b797ef39c2f693554821ec7bb3d96689661))

## 0.30.2 - 2023-02-15

[a0d6266](a0d6266c487f1ccafa44516c001eef027766ad98)...[f6e8c11](f6e8c119ff366f3269bddc733f9c9f4d82167693)

### Chore

- Remove unused dependencies and update poetry lock ([7b4b658](7b4b658c2649500809ade7efd716e8dca4153576))

### Fix

- Ignore single import mypy error ([dd85159](dd851590719ec76ab66dc9d908493991fc235e7e))
- Use non-static path to example folder ([613a152](613a152e65fbe0f7116a1f772fea8a3836d888af))

### Refactor

- Remove deprecated examples ([eec3f0e](eec3f0e75a7875a8a2d1da9c2ffe586a4a18ebf9))
- Remove unused module ([d2e643b](d2e643b1368a5776829a0353730afa5039c19590))
- Move test in the unit folder ([89df933](89df933b50eb35e0528042f81a37a59ba8630ff5))
- Create integration test from POS tagger example ([cb73343](cb73343957c6b75df2a741b08c66c11545b86f2d))
- Remove non-deterministic test ([ebed2a7](ebed2a73beaba1f9e6abdc843eb5771cc1d34061))
- Remove deprecated example ([008241c](008241c8d5414cbe9478e1cdb226c22c48b2c663))
- Move tensor_test_case in tests directory ([3cf635b](3cf635b2d5ecbad524cfed75d4d4b7543c2dbcc2))
- Delete not relevant example ([3c0fce9](3c0fce95db8c078b8e37e34d0018872164402c4f))
- Rename example ([84d4792](84d479296c1930f4e7f334ae1d2fd89ba84b595a))

## 0.30.1 - 2023-02-04

[a6f9577](a6f957796c627bcf7af43d78cfdc767331d6092a)...[a0d6266](a0d6266c487f1ccafa44516c001eef027766ad98)

### Chore

- Remove vhdl scope ([5c9571b](5c9571b384588551c7439f3e45ad63d8f718b79f))

### Fix

- Make test more deterministic ([97fd410](97fd4101af93cf17d446cb0cb38a419080d5bee6))

## 0.30.0 - 2023-02-04

[46f013b](46f013ba3fb9e90856882296095a011c00457ad8)...[a6f9577](a6f957796c627bcf7af43d78cfdc767331d6092a)

### Chore

- Relax commitlint rules ([108e361](108e361f763f23843b72c5620cbebd0c171a9433))

### Docs

- Add commit types and scopes ([e759fd3](e759fd38fb41d413ccf03617f84f87f6df9aeb12))

### Feat

- Add unit tests for the LSTMBase layer ([589f803](589f803fd858b22985485d795f4441a9abf97742))
- Add unit tests for the fixed point quant/dequant autograd functions ([f82431c](f82431c164b9536899d0cca9b391a057add8187a))
- Improve TensorTestCase class ([d4273a6](d4273a60c169669ddba5f80636d1430b69c77d90))
- Rename quant_typings module to quantization and implement FakeQuant ([0e5f24a](0e5f24aeb9f43258f9e971ffa777c585faff05f0))
- Integrate arithmetics for the linear layer ([a961558](a9615581159ba4b962fac8458d9b76de0a61d98f))
- Convert example parametrize_convolution to automated integration test ([3dde1c2](3dde1c250fa4ebb617bbd543c9b26cb320d430f7))
- Convert example translate_linear_model to automated integration test ([5d92d0b](5d92d0b15d8c0a1d76f842fd7a8bbc591bd1cf18))
- Remove input_quant and param_quant and add quantize function to arithmetics ([ee91e42](ee91e42801b0d1163a0d52130fc578477da60c74))
- Implement concept of arithmetics ([e7ad504](e7ad50471e2ac7300e0db781bd37cbba1364a5e6))
- Remove quantized_forward function and adopt tests ([c865c73](c865c73a53e89c40ecebc9c4b49ba6d5c14256c1))
- Add example to demonstrate that the new kinds of layers are trainable ([231e325](231e325815c469596c63259c5f345dc9afb0f3b7))
- Lstm uses fp hard sigmoid ([fd265ac](fd265ac3e1ef7f11e28236705e4a38760462bddc))
- Integrate hard tanh layer ([eb74d3a](eb74d3a3671616db37ba8f554332ca1ddc33dffe))
- Small example for translating combination of lstm and linear layer ([12e7101](12e7101e8c62e8424bc2ed580cfbe645e8d33510))

### Fix

- Fix imports and use new FixedPointFactory features ([e8c74c3](e8c74c34ec1c5a4b5189d74f2a19a993a5ae9779))
- Adapt basic qtorch example to recent changes of the creator ([a17d900](a17d9006240a67da97b8a539620aa1974e07e942))
- Add similar concept of translation arguments to fix the translation process ([e387ae2](e387ae26918fbe8e4a0ee01ccc4361849746bd66))
- Fix unit and integration tests to use the new layers correctly ([0553017](05530178cf7fb64dc88cab82b89c24b2a1406e8d))
- Remove unused OperationType type and FakeQuant class ([596dbd8](596dbd8cdf3cde67eedea2779a35ff682c9ac9f7))
- Change torch LSTM layer to our FixedPointLSTM layer ([5e7a39a](5e7a39a78684c09a1d374476f8fb611019ae994f))
- Infer fixed_point_factory of linear and lstm in build functions ([81df686](81df686fe13db5f85c91b65c73713b7da8e6c64f))
- Fix LSTMCell raises Error for unbatched input data and add a test for this case ([5ce3e21](5ce3e2125b4bcd1115d77ebe5c833e52d58bad77))
- Rename to .tpl.vhd ([fe3c85c](fe3c85cd77d0f2fefb90f2d3ff6eadde8570d000))
- Remove sigmoid_resolution ([dd4f033](dd4f03366920f1a3774772a16a49efaa8756d249))
- Use model.children() instead of model.modules() to avoid recursion ([a3c349b](a3c349b13af0fef383b494850973d8ff9ac2dd68))
- Fix some mypy errors and remove unused imports ([08e2362](08e2362fa32efd13e388140ad58c93b0e79229b3))
- Change not existing layer_id field to layer_name ([f7425c5](f7425c515395243962db1517116b9961b1668cd7))
- Add layer_name to all vhdl templates and components ([2d9c47d](2d9c47dc60642d94efeb58cc3014f6a7790a6f26))
- Fix errors in the lstm template and remove lstm_common component ([c4a28ce](c4a28ce2f40dc84e7a5e4470c62a40911b73901f))

### Refactor

- Move unit test to correct location ([c03c362](c03c3621c6cdef58a44e1c3e279d025ebdf34aa6))
- Remove unnecessary print statement ([2f8a0a7](2f8a0a75b602d6d7621f310e33ccf0bf0d5c1e28))
- Remove examples belonging to the removed precomputation package ([4dc681b](4dc681b18207dd92d767c97df2c70e2fd3e6cd2e))
- Move integration test to more specific location ([0115399](01153996ac556eb9a96f404e8efed2af5bbdf1dd))
- Remove default bias value from linear layer ([8d55471](8d5547180a50f07ee259f37cd8cd89ffe496e421))
- Add more precise type annotation ([0c47fe0](0c47fe0b485cb71662ef017b7c454b848baa0b4f))
- Remove outdated evaluators ([8c0009a](8c0009ae54dfed9f24223ca01a6b146ee0c06f04))
- Add fixed_point_factory property to fp layers and remove FixedPointLSTMCell ([9f0a5d3](9f0a5d3505dc05d53aaf9fa9fb1c607049c661fd))

### Style

- Beautify commit # [6209df2](https://github.com/es-ude/elastic-ai.creator/commit/6209df2bbc3c693f1829ce8b93822fc84152f69b) ([423b081](423b081476868df0a7f90fbcaeec16203670551f))

## 0.29.0 - 2022-12-16

[58f37d2](58f37d22b9b09eb48c2b3a604d3365f7520653e8)...[46f013b](46f013ba3fb9e90856882296095a011c00457ad8)

### Chore

- Tighten commitlint rules ([47a35da](47a35da220ba1c6081af11b0a6e7945978f2fe77))

### Feat

- Set pypi project api token ([37ba8c9](37ba8c9794acc6b4bdf64087c98c61172446fcb6))

## 0.28.0 - 2022-12-16

[6346bfe](6346bfe40ac06023434683c8ec2e8d73e9e246ed)...[58f37d2](58f37d22b9b09eb48c2b3a604d3365f7520653e8)

### Chore

- Use gh-action provided by python-semantic-release ([0d0321e](0d0321e44455d40c3b04929df13cccfe7056c35c))
- Add noop to semantic-release and trigger on workflow call ([ecdb463](ecdb463514c0e5b8b0d0d22818071c728e6997e2))
- Add github action for test environment setup ([8a38722](8a3872210155601a900b8ac59757808974961999))
- Rename actions yml ([7882524](78825240f3cd78863110f516574d915781f3a4c5))
- Add commit hash to action reference ([459a4cc](459a4ccd2487762c67a1be86f2ae071dc89396e8))
- Fetch repo in job instead of action ([05d8bd1](05d8bd14a7c287c90755ffb68f2c899d3d182ad2))
- Specify shell in gh-action ([a5fb59e](a5fb59e35e8b557011559ba5d55b68a452574710))
- Create cache-dir in action ([f0ecc17](f0ecc17eedd1e9acdc6c0d4baa713eee6a5e2495))
- Reorder poetry calls and cache setup for action ([2a0fb0d](2a0fb0d65d5bf80746b65d1d5f29f63cc59f36f1))
- Add missing argument to poetry configuration ([1567b0c](1567b0c7269f14a1454b206b959c2c33862fe239))
- Enable semantic release for main again ([6c93920](6c939203995883b390a20bc98b098a252563c669))
- Temporary relax commitlint rules ([437c3d7](437c3d7cec0487f5754ec357fb4d313343fd2cbc))

### Refactor

- Remove unused import ([14d1d60](14d1d60bb7b56c2c6bdd00feb767a8248a09699c))
- Rename package from qat to nn ([e211ae6](e211ae63d9ee7fdc2c0fad15a40730399fac7654))
- Rename _init_quantizable_convolution function ([2b57dbc](2b57dbcaa02f202c7654d8d15b53c84a0210ee1f))

### Revert

- "chore: add commit hash to action reference" ([e42d010](e42d01029b403029334dc2ed1a3311631361f9fb))

## 0.27.0 - 2022-12-15

[913658d](913658d4df2f93a54270d7591b974a65fd34b34a)...[6346bfe](6346bfe40ac06023434683c8ec2e8d73e9e246ed)

### Chore

- Set correct path to unit and integration tests ([538eb2f](538eb2f036f24ea99135f0e66ad59c3738e60231))
- Remove superfluous line ([71edbc4](71edbc4369a59c90c561ff3e8b335bd85ecbba7e))
- More specific commitlint rules ([bbb88e9](bbb88e9080ecd873209f99aa01473b9d57bd2012))
- Update poetry.lock ([0f78c4b](0f78c4bfdddad038bd69b1a92f3b1fba4c5ab9f8))
- Don't install extras prior publishing ([effa8c0](effa8c004a2d8356a96e3869763e85e58ee92924))
- Tweak pyproject and commitlint ([addc521](addc521744804fb8a6deeadde8510bd9fe37d87b))
- Add style again to pyproject and commitlint ([d7aaf28](d7aaf28042881c272f851e5402135d15a149ec42))

### Ci

- Add commitlint constraints ([d345351](d345351c96c0ce6d0a5bcf52ae9bca8eacdafd6b))
- Adjust commitlint config ([e518975](e518975728140ab29692bea341ea015cfcfb59df))
- Temporarily disable commitlint constraints ([87eaa63](87eaa632c644d70c5ac693b2f5f6aac6a3625acc))
- Temporarily further relax commitlint ([08eab5b](08eab5b817d772457dfa983846adb36a8f1b64d3))
- Don't install onnx ([7b164cd](7b164cdd95a5af672f78c7c22e267c3364fe4d0a))
- Clean up pyproject.toml ([547d724](547d724db5c14392b148c8cf9e0a5714b1052a4d))

### Doc

- Add doc to VHDLFile ([5fcf78b](5fcf78b87edf75ff3e9e818b1511aef00ffbf46a))

### Docs

- Move tests and remove deprecated lines ([4a074a8](4a074a87fb31df535d415c2ab6aede7e4d7d8949))

### Feat

- Add constraint type ([dc4c4e5](dc4c4e57a9615a9be6941ecc750d3838458ff919))
- Update qlstm sine wave example to the correctly implemented QLSTM layer ([dc62cd2](dc62cd2aa05067b164009301ab7c5e110797c503))
- Remove constraints ([6b7b483](6b7b4835dc9f9f6b6fc83bc619727aa948c19161))
- Support generation of layer connections ([1d43c42](1d43c4212ef54c5488df7e7dc3829df31a7e8484))
- Generate portmap output_address ([c6a26a6](c6a26a61d98c90fa29b02e6619116e67a4a67ac5))
- Add hw equivalent module tracer ([3f2c2c7](3f2c2c7acc5046131d420d513a4bb3d3981ac0c5))
- Tracer records reference to module for call_module nodes ([20ed7da](20ed7dab9677e476925a8b1250cbbc2004d43246))
- Generate vhdl signal definitions ([53408f6](53408f6cb9daa5c44931e880fda0712c2924b822))
- Generate layer instantiations ([7a75fc3](7a75fc31780a6173424ffdcf3129bc60d5a83e59))
- Introduce HWBlocks ([ab03eaf](ab03eaf28c74483fcd9dbd78d247d39e248bdea1))
- Extend code file with parameters ([4833f8b](4833f8b2d5553cf02d322b8485587612cd67a9e8))
- Implement HWBlocks interface for sigmoid,linear ([0177373](0177373eeddfa9c32100777bbcd7a94765dc1122))
- Add module_nodes to graph decorator ([6d0a612](6d0a61217b36b9db8e9df19210e5f0d3aeed4ef2))
- Introduce HWEquivalentGraph ([844bb84](844bb84a2d36e50f3de7ae4b713d370011d3240e))
- Introduce HWBlockCollection ([a80bda2](a80bda2d705992030b18649ff99f3a6ce75d7ef3))
- Distinguish x/y width ([2f52100](2f52100d32502520ce66a240bae90dd48e070ebd))

### Fix

- Fix circular dependency ([1d5615b](1d5615bf81757bf16904eb75c33fead69a68dd43))
- Fix the problem of wrong shapes for the QLSTM layer ([b75f478](b75f47804016a3dfdad3f8d2dd575f4252cac5ff))
- Fix error when passing flat input data to _QLSTMBase and batch_first set to True ([29918d1](29918d11c508e3e91fe00a0e07988be0ed198b35))
- Remove unmaintained onnx support ([dc773d3](dc773d39fe2c0ea5785e3fb0bf7a43f3bf83495f))
- Remove obsolete vhdl formatter ([83d81e3](83d81e348152e047482ccc45a2ccaf6173f772d9))

### Refactor

- Using code function to generate code instead of call ([843ad64](843ad64d33e2018da9c88fd487ccd46fd598c58f))
- Add missing type annotations ([c83a746](c83a7466cecfc043dd95800c69b6ee5df8b5bd4f))
- Add missing type annotations and remove unused parts ([6fb622b](6fb622b3cff4caf8d849c8df8696275fa38fa9bb))
- Rename call function to code in the language module ([4cf795e](4cf795ee1cc679ea6b4b7cf51198cb536a5d9af5))
- Rename call function to code in the precomputed scalar functions and test benches ([a40553e](a40553e05f64d9bd57473fed4d40b269858ef65f))
- Fix some mypy errors ([cd9899f](cd9899f55476cd91993f7276cdda02fc7e3d7b26))
- Move all unit and integration tests in a tests folder ([8afb751](8afb751a5dc9f7fd4e2fa4a1dd1167682efe590f))
- Move BatchNormedActivatedConv1d from layers module to blocks module ([6269522](6269522bcdb978a905c84693e6c9fa4bdc32bfa7))
- Split LayersTest class into classes for each layer ([55c12b3](55c12b36ce0b9807ffa4f5dd8344e3b8143f1212))
- Remove unused import ([b6cf349](b6cf3494b36cca9d2fd732a24952423b68ad6c46))
- Remove unused code ([43e5992](43e5992f1e48d078113bba7863c0ac5e3e967ada))
- Remove unused code and make Identity quantizer public ([dcd726e](dcd726e183c5b74b05c27155ec64cc08f395802e))
- Remove noise comments and remove default quantizer from QLSTM and QLSTMCell layer ([4a57ca9](4a57ca900a6c5dad1710f6d558c1ade17527d2b4))
- Remove default quantizer from QLSTM and QLSTMCell layer ([cce2f8f](cce2f8f1d22c384f136583f74d3a2b396500b0e0))
- Create a _QLSTMCellBase and _QLSTMBase class to avoid default parameters ([98ba0b7](98ba0b78090c120070e38bf9b0502b3027e0fa33))
- Remove noise comments ([ccc3979](ccc397911d899bdc49b917d0438336e17e37d100))
- Reading text from resources returns list[str] ([361c5a5](361c5a571432f24b8b2be0327fdc1b4edea1c6fe))
- Use streams for reading templates ([2af5d2a](2af5d2a41d72406a4abcbb2c55f3c0c01150cad4))
- Remove CodeModule/CodeComponent ([89e27dd](89e27dd4e1bd77502d69054c15a3277f0a4b0826))
- Use new vhdl file class for hard_sigmoid ([a9e1f6c](a9e1f6ccba2f978b8290be94c754872432d3c311))
- Refactor FPHardSigmoidFile ([3c54418](3c544184877d3416445f064d33bee3e95d78ac31))
- Use VHDLFile for root module generation ([e2b8423](e2b842310f237ff8802bd92ac4f7f537d9ede707))
- Move files ([01999c4](01999c4049f21e357390c2fc88a09bfc987d0cb6))
- Rename BaseHWBlockInterface to BaseHWBlock ([4c69682](4c696826e559603ac54681eae7ff50e34d22a1ac))
- Move classes out of hw_equivalent_layers.__init__ ([0713211](071321164dee3e9a9db3d113848c6ce3dd960b1c))
- Sort imports ([2c114f1](2c114f1b8c839ef939051f5a1c5b6d40585908cb))
- Move files, simplify, correct types, merge ([57d3754](57d37541fed53c29ad9e6a665aa72b99fe5a2df0))
- Move last tests out of old test folder structure ([a3e12c1](a3e12c11df45b4de8babb2f1862ea92a1778c92a))

### Style

- Remove deprecated code and move/rename ([f6b8020](f6b8020f8a5dfc5d9226578efa5f2512b84223e5))
- Introduce template code file interface ([69fb2b6](69fb2b681497289d230f8d203f5f430c91a3ff54))
- Beautify commit # [5fcfc23](https://github.com/es-ude/elastic-ai.creator/commit/5fcfc23c342983a98efc1d527648ef17644c472c) ([a228cc0](a228cc00bf0a44327a858835e9a73531af56e59e))

### Test

- Start splitting large LayersTest TestCase class into smaller classes for each layer ([905f165](905f165fabf01e0a5897ffc41bff01acac3175b2))
- Start making network.vhd tests more specific ([ea260f0](ea260f0bc989a0bc477b425f14364bdb7756a37b))
- Add multiline begin-end extraction for code ([e958557](e958557fccaabf46d4020cf23fae14e20c4452ee))
- Introduce more fine grained testing for generated vhd files ([5a27cdc](5a27cdc16ed7b3cbb880c022ac65e1adcfdca563))

## 0.26.1 - 2022-11-30

[dd5511f](dd5511f9df92c33e31c8449f7e9a957a206e93c9)...[913658d](913658d4df2f93a54270d7591b974a65fd34b34a)

### Feat

- Implement quantized forward function of the fixed point lstm cell ([7818e15](7818e15bc6c41454090b77fe5df7a8e7930ab570))
- Start implementing lstm base module ([b154ca5](b154ca5525c00f735150c21f64324da87328ba5e))

### Fix

- Remove layer_name parameter ([7a83b1e](7a83b1eed3095a8b7f90438c78ba24bba6e44958))

### Refactor

- Rename lstm module to lstm_cell ([97bf791](97bf791a8a3fbab68179f5a9a20e9410c3bcccf7))
- Remove examples that are not relevant anymore ([3b241e2](3b241e2ddfe14a248e411f0b8da9ec6cf85cc8bc))
- Remove vhdl formatter that is not used anymore ([007a8c4](007a8c4ec4c42382390b0af034a2f5f3226fea86))

## 0.26.0 - 2022-11-23

[d1a9b56](d1a9b56c4661bed43895e83b54051f9888a1d634)...[dd5511f](dd5511f9df92c33e31c8449f7e9a957a206e93c9)

### Chore

- Remove minor python versions from gh workflow ([fc517a6](fc517a6bb81fb037f2b9d3466d32506aa0573020))

### Feat

- Merge from main ([fefd3ba](fefd3ba4ab1fa8ae9d09bfc6185f906175f7a6ff))
- Make linear layers better timing ([1c6a3ae](1c6a3aeeeaee929affbb092eb485c1cf7a323355))
- Clean the code ([d737d02](d737d02122207bcd24f4b7c960b71db095d34a26))

### Fix

- Fix error during integrating to a MLP model ([0e2b89c](0e2b89c898497f35a2ad840bd3065429799bdf61))

## 0.25.0 - 2022-11-22

[0c8fdc2](0c8fdc25eb6977b2bfeb0d9523efa11ae8167f08)...[d1a9b56](d1a9b56c4661bed43895e83b54051f9888a1d634)

### Feat

- Add expand_template function that fills string templates instead of format strings ([eb9ee98](eb9ee987f73ffb26e8280ec3c32b32e38896d3c1))
- Apply the expand_template function to the already existing templates ([c958f54](c958f545f4c2cf2414a007753b416ec73c410458))

### Fix

- Remove the layer name in the example file ([767b5f9](767b5f9c62d493d35e5a294b1363c861d5438fa5))
- Fix small error in the template file ([fe94518](fe94518ff2e5e44f7c1ff8f9bf8b4ff8f0b5cf41))
- Fix the error from merging braches ([c386766](c386766ea654852c5ad5254cefc1fab28f544c66))

## 0.24.0 - 2022-11-22

[a7eebbb](a7eebbb669e5329ba9731b0282e3e2a2eca612a6)...[0c8fdc2](0c8fdc25eb6977b2bfeb0d9523efa11ae8167f08)

### Chore

- Update action versions and remove usage of set-output ([d106116](d1061167f9da09f7aa2a191340652ae56e3335e0))
- Set correct commit action ([a7a2439](a7a2439d8b8a1358983d18c22de6e57820757d82))
- Set correct parameters for the commit action ([373ffd2](373ffd20d331152cedde6083d72fdb40d83c741d))

### Feat

- Implement FixedPointHardTanh layer ([ed72810](ed728101fb596a08e1a76d936d04306a066c50b5))
- Start implementing lstm base layer ([39ce891](39ce891d56be59d5a20a36889b0e9c2f13e00bd1))
- Implement and test lstm cell base class and start implementing fp lstm cell ([f458fb6](f458fb6c216385a119774a3f98788941e13ed5c9))
- Add layer_id parameter to build function and set it to a unique value during translation ([cfdf949](cfdf9492190e24230293e3b0b1b312bfc9710952))

### Fix

- Fix wrong return type ([eb53ed9](eb53ed972ec9078f6c405ecd7c92043eaf8ed419))
- Remove duplicated key ([5a4bcd6](5a4bcd6fb6de9cff6c639866db1dd50918f3039b))

### Refactor

- Move common helper functions to a separate utils.py module ([b459f0a](b459f0af4b8b9884c03277928d7a0437b68f9716))
- Move OperationType type in the typing module ([bf0d3fe](bf0d3feacdfdf461527056b8a24aec63907a2578))

## 0.23.0 - 2022-11-15

[c66096a](c66096a44b6a99d090c2f0754d893ec6737fd100)...[a7eebbb](a7eebbb669e5329ba9731b0282e3e2a2eca612a6)

### Docs

- Change documentation ([d3fb540](d3fb5402c7acb09cee3df535671f22d5011f2f47))

### Feat

- Merge main to current working branch ([35db3c5](35db3c56608493c6b33d05e0c2250cedb0374c8e))
- Enable multiple linear layers in the same model, by adding layer_name ([3a99a30](3a99a3059dd53b913e7d619cbce28014007bf854))
- Remove the previous linear_1d implementation ([0f1b9aa](0f1b9aa2f1c12f5c0fc1fe6a3db482f40041c057))

## 0.22.0 - 2022-11-13

[518bbb6](518bbb6fcc78c870f9cf82133cae298e12d918c1)...[c66096a](c66096a44b6a99d090c2f0754d893ec6737fd100)

### Docs

- Add missing parameter in docstring of the translate_model function ([458a02c](458a02c38402a0860500d5821b68890fcc78c01a))

### Feat

- Raise an exception if the build folder already exists ([d09bfa1](d09bfa105d909b58432cf8883ee55a6b11639add))

## 0.21.0 - 2022-11-13

[f71ab12](f71ab12692dbd71a8227087c1c4eb885f8d674aa)...[518bbb6](518bbb6fcc78c870f9cf82133cae298e12d918c1)

### Feat

- Add fp_linear_component and its template unittest is passed ([6e97316](6e973168ca244e4cf407c48b31406d2eed73b4b0))
- Add fp_linear_module and test passed ([241fd65](241fd652495d6ce582873f1bcc297302f3d61764))
- Add fp_linear build function, and test passed ([ffcbb1d](ffcbb1d57408ad03e91bd1228bc6d3289f1d0c66))
- Add default build function mapping and small changes ([b1d6f2a](b1d6f2ac1040e63781d5f4af7ee29e486d9b6d69))
- Check the component interface ([53791c5](53791c5eb9a72793b16a0a41eb79ed8932b8e32d))
- Add fixed point relu to translator ([80935ce](80935ce550a2e99267a55b41ad272906faf211a5))
- Add default build mapping for fp_hard_sigmoid and fp_relu ([c9c4d9f](c9c4d9f329ed2c56d47f2b698dbe1d3b34c1c8a5))

### Refactor

- Change the interface name of the template ([a693041](a693041a050ef77828a4f4dba791b0a38a845184))
- Get rid of some comments ([ced1b12](ced1b127031c02d8576ccc35fdd9f143017c3368))

### Test

- Add test coverage of relu component ([88e0d10](88e0d10d4cb0a64ac397cee1a9e42db9184c6139))

## 0.20.1 - 2022-11-10

[51dd88a](51dd88a341570fcd4eeaab201c112ba44c43de5b)...[f71ab12](f71ab12692dbd71a8227087c1c4eb885f8d674aa)

### Fix

- Fix incompatible signature of the forward function ([ff6c165](ff6c165cd0bf17477051548018b791809fff33c9))

### Refactor

- Small change of the FixedPointFactory type ([c7629bd](c7629bd05764de09f03fd8445437dee671518d38))
- Remove usage of deprecated assertEquals function ([6a6f4f3](6a6f4f3af28735e27fc70b51f857324cd1ead7ef))

## 0.20.0 - 2022-11-08

[7568be6](7568be66864623be1b31c27db21573fb18efe0e6)...[51dd88a](51dd88a341570fcd4eeaab201c112ba44c43de5b)

### Docs

- Add documentation for the quantized_modules package ([9da4a0d](9da4a0d380304a7ab8834049ad93bed547816ddb))

### Feat

- Add example using quantized modules to verify the current state of the translator ([0c55e00](0c55e00657c0d260766155995b75f25bff642e24))
- Integrate fixed point hard sigmoid to the translator ([0a07cee](0a07ceeb3d238456dad08448b543f4a075873322))

### Refactor

- Change name of the output_quant parameter to output_dequant to be more precise ([3186b5b](3186b5b848e4b7be8e0bc0d94a1897722d2e2397))
- Rename example to fit its actual content ([9ac5a83](9ac5a83a558887d0cf4830a6f7ba94ede92de594))
- Remove unused import ([fbb684d](fbb684daaeb8376ec7a56b413959cb9e9f2dc600))

### Test

- Add tests to test the quant and dequant parameters of the LinearBase class ([ad49bc6](ad49bc68ef5f38e6047d569d8e90513f50698a27))

## 0.19.0 - 2022-11-05

[51f771e](51f771e64860e869cf9f35ffc21a027e4d1a0f72)...[7568be6](7568be66864623be1b31c27db21573fb18efe0e6)

### Feat

- Merge translate_model and generate_code functions ([c12562e](c12562ee4a55c61b5ef82b5ef37568fe32e8f525))

## 0.18.0 - 2022-11-04

[bba0d93](bba0d93501bfad3cc38a640c33c0afbc71f7c7f6)...[51f771e](51f771e64860e869cf9f35ffc21a027e4d1a0f72)

### Feat

- Refactoring and start implementing hard sigmoid activation function ([ff94c9d](ff94c9dd1d1297f02e82a0d1f7f203f80c8d2732))
- Fix wrong calculation of fixed point values and add quantized forward functions ([93046d3](93046d3b93d1a977c4106cf56e7f98847a47aa00))
- Implement a version of relu for qat and quantized inference ([ddd9607](ddd9607e8dbf333817112dfe24f795ac717f609e))
- Use fixed point hard sigmoid and relu in the example ([90350b9](90350b91b9ac917c8c1f0ab50c2744fb09671947))
- Implement evaluator for simulation of a quantized inference ([353e82e](353e82e798359c3b15a42a02dcdc63e071b2d34e))
- Implement evaluator that evaluates a model according to a given metric ([a0b089a](a0b089ad1f7c32acc0c4522bf830080442e8414d))
- Add simulated fixed point inference to the example ([4f81d8d](4f81d8d3d44f1c677fc1a12edf94b7b614d72efb))
- Add clamp to min or max fixed point integer for overflowing values ([ca3fc19](ca3fc19aec062d4de34a4698c9e0a9351b41c761))

### Refactor

- Create a better module structure ([b2dfeee](b2dfeee795d980aabc2822e3b1470f2e41d63416))
- Removed unfinished fixed point configuration finder ([fa6dc44](fa6dc44e0a02f57993f08b381b62297c2682b167))
- Small changes to make the code easier to understand ([7168ba1](7168ba145243616d247006f56d99de3c21e91401))
- Make floating point values more explicit ([231b903](231b903127dcdc0b90bc6a4e29ccd29543033935))
- Remove unused line of code ([f020554](f020554dc3f1ae6ed4ac025711e6ec1025ba8964))
- Rename fixed point linear layer example ([59216da](59216da6973daca87e80c105513586df1c682ba6))

### Test

- Write tests for the evaluators and do some refactoring ([2641578](2641578eb1820793e4e2117563dc11607707e11d))

## 0.17.0 - 2022-10-22

[0ba60d8](0ba60d8bdc85599868bbb96281d94acf2d47b39e)...[bba0d93](bba0d93501bfad3cc38a640c33c0afbc71f7c7f6)

### Feat

- Visualize model parameters ([5e1b4fc](5e1b4fc4c827c55d19cb9bc4206f706bcc737fba))

## 0.16.0 - 2022-10-22

[01d8c35](01d8c3518096459326be48435e0df35b4960e105)...[0ba60d8](0ba60d8bdc85599868bbb96281d94acf2d47b39e)

### Chore

- Skip some tests that are not relevant at the moment ([2dbfd55](2dbfd55a805166e97b640fafd5ebc1214288a863))

### Feat

- Add function to get attribute names of an object matching a regex ([acc8e29](acc8e29e2771d5642e1371af6fb3c44f83b5ebc7))
- Implement custom linear layers that allows to do fixed point calculations ([c2364f6](c2364f6182bb8406e90a78d632bc868537705fd2))
- Add a type for fixed point factories ([53b8499](53b84991671832c2e7fa24e61d927b7c039832d9))
- Make base linear package private ([b3cfa55](b3cfa55daff5c401bc036ffe2bba8b0c6b2f2554))
- Move tracing example to example folder ([b942155](b942155a240a6f34f0f02361b6631b431a448443))
- Start implementing an example for learning a simple logic function ([6cff6de](6cff6deccd5c2080e930d93f5e145e4d7ea6a41e))
- Add feature to automatically derive fixed point parameters from a factory ([70618d5](70618d512718efd7e718491af52e1acbc6c86622))
- Commit current state of the fixed point linear example ([9b8ecae](9b8ecae971bc1dedabf17e79272008a3cbfb5123))
- Move the input, weight and output quantization to the linear layer ([0c8b259](0c8b259ef688c606ebd4f8486ef7b6f48e0f8713))
- Add the ability to plot the model parameters ([b1b0b5e](b1b0b5e7697992c4c53825c739e2fb2dcc903dac))
- Implement qat for linear layer ([d3ba49e](d3ba49e266b2931c1b16677dd91f17a75f091501))

### Fix

- Fix bug in the linear matix multiplication and rename _BaseLinear layer to _LinearBase ([da11356](da113561d69158ccc2a9266adb1eddcc79b1cb7d))

### Refactor

- Remove unused typevars and change typing ([2770991](2770991c00a9395697180fcec98e733164efde24))

## 0.15.0 - 2022-09-29

[393d96b](393d96b92bba7c9c47f8ee222e2dda619ed40259)...[01d8c35](01d8c3518096459326be48435e0df35b4960e105)

### Feat

- Implement clipped fixed point representation ([8e53506](8e53506fce0ba5adaa124ccd61de3b340bf1c95f))

## 0.14.0 - 2022-09-28

[5581b87](5581b878ae1302451ab51af81cecee3d6e9c60ed)...[393d96b](393d96b92bba7c9c47f8ee222e2dda619ed40259)

### Feat

- Implement automatic derivation of fixed point parameters in the lstm example ([504008d](504008d7ef3f402f8476bb77f02a4a37176d229e))
- Implement signed fixed point integer to FixedPoint object ([0a2fc79](0a2fc7952dc13ea48c749856bf809a5540166598))
- Start implementing fixed point evaluator ([0f9c62a](0f9c62a38f9df6ee4e84f1a3b5524df03511b438))
- Working further on the fixed point configuration finder ([beb9da0](beb9da0ec8c3fbc6bb4ff65a97e7424e4da6dd0d))
- Implement from_unsigned_int and from_signed_int function and remove unused function ([aca77f5](aca77f5eac396f21821b07706ff250b2589dd037))

### Fix

- Reimplement unsigned_int_values_to_fixed_point function ([cdd069e](cdd069e6adffa882bb34fea2b7179891c282045b))

## 0.13.0 - 2022-09-10

[653835a](653835a3ddb383bfed60613928eab732b5f93855)...[5581b87](5581b878ae1302451ab51af81cecee3d6e9c60ed)

### Feat

- Remove translatable protocol ([37412e8](37412e87d89d16c9159cf12ef00032343119100c))
- Explicitly set poetry version ([82d202f](82d202f0229e7931fc7371f69abe0d1fe3a58134))

### Style

- Beautify commit # [2024c7e](https://github.com/es-ude/elastic-ai.creator/commit/2024c7e9a8aa9ed2487f58586aa41beabd6f63d2) ([7a58041](7a58041b71ccd258d9fcb16b1ac1a15be32e212d))

## 0.12.1 - 2022-08-29

[02d0ba3](02d0ba3850d892626ee39e0adb31939605b5836e)...[653835a](653835a3ddb383bfed60613928eab732b5f93855)

### Fix

- Reimplement binarize ([9bbccdd](9bbccddfc6ce6c2b928166cdfaf1112b294dba17))

### Style

- Beautify commit # [7a60a04](https://github.com/es-ude/elastic-ai.creator/commit/7a60a043e83aedcdf281ec9357ee9f274aca59dd) ([0723cb1](0723cb12e8fc6290403efd68b6d552a81ad69a99))

## 0.12.0 - 2022-08-26

[ff37f26](ff37f26d0f570d154e912c51cfffe84a755484fd)...[02d0ba3](02d0ba3850d892626ee39e0adb31939605b5836e)

### Chore

- Add a dockerfile ([e2f54b3](e2f54b373bf26ee6c94b0c0c448b6e34affb2e64))

### Docs

- Update documentation according the newly added linear1d layer ([41e2486](41e24868aecbf310ee4c9ad815f6ccc0da3f9f9b))
- Small changes of the documentation ([9e7699c](9e7699ce617581f67f85cf4ef7d945d99df241be))
- Move translator documentation to the vhdl package ([9a90949](9a90949528978ff4732f585986a71cedd44e82a5))
- Adapt diagrams to the latest changes ([c1750eb](c1750eb19f92a705f8f36ccefc9729d3545f0743))

### Feat

- Insert values in the updated lstm.vhd template ([4d9dccb](4d9dccbdb11afebb466f476c22539828bf5458b1))
- Make work library name customizable ([95fd8aa](95fd8aa0d7e512aeb04893de2df2e58cc4b3e641))

### Fix

- Fix calculation of the addr_width of the linear1d layer ([6fa2b2a](6fa2b2a3bc83d3a51eb955d1464501662f6676a8))
- Pre-add input-hidden and hidden-hidden bias ([750941c](750941c3150cabefa2f393f6b12105a358a70f7f))
- Add changes from Chao after testing the translator ([5a5d532](5a5d5325a3f598e0163d4eac0601b5961c2f5780))
- Fix test ([c125bf1](c125bf16297ee9e39660ee904ab54268e8901d48))
- Remove unused work library ([c68fd9d](c68fd9d00b152c5bdb70d2d2c90ca8d3e9f381d0))
- Remove some comments ([13cc1a1](13cc1a1ade14ccc7aa686523270dec20936ed14d))

### Refactor

- Simplify code and reuse components ([37cffd7](37cffd76953e1f7756de8ec7ebc5b356fb89f1ad))

## 0.11.1 - 2022-08-18

[9f2babe](9f2babede009ac27b1587175403d8fe10c735f16)...[ff37f26](ff37f26d0f570d154e912c51cfffe84a755484fd)

### Build

- Perform tests with more verbose output ([8d7b50b](8d7b50b7ae2c6d513f67027e5f209cf3115e0964))

### Docs

- Add documentation on how the translator works ([91ebea3](91ebea3fb7e7883f56b2cd9152769d151449a49a))

### Feat

- Implement the translation of a linear1d layer ([b627e78](b627e780d054adcdd89009d87aa33fa31c913504))
- Add an easy way to get a fixed point factory ([d98ff03](d98ff0351f739859ed668a2ec295421e29fd24ec))
- Add linear layer to the translation example ([5f1e1db](5f1e1db8da7ce533cb592d56ca97e25ca563a60e))

### Fix

- Remove deprecated threshold and codomain properties ([5db9669](5db9669fc3942851e65607a869bb822430df7836))

### Refactor

- Change naming of the translator components ([fdf5586](fdf5586da727542be6bfab57fba4a98d8ec482d7))
- Change naming for better understanding ([17d8a3d](17d8a3d89dcbbb4882c62953ddb928d268945852))
- Small naming changes ([fd5c9b4](fd5c9b4f9fccb95b9fd4a0223e87a791fd02224c))

### Style

- Beautify commit # [52e7e3e](https://github.com/es-ude/elastic-ai.creator/commit/52e7e3e55053a9e95e786bf899056148753cddfc) ([a5c17b4](a5c17b428c20f8c55b7c9350e5d9a33ef8b76822))

### Test

- Add tests for rom and linear1d component ([fc1f20e](fc1f20e30bf91f6aa96ee800e2e66aa5b3c217ad))

## 0.11.0 - 2022-08-11

[0288e04](0288e04e546dbc91d60884588fe5a31e0f81fa7f)...[9f2babe](9f2babede009ac27b1587175403d8fe10c735f16)

### Docs

- Fix commands of install dev dependencies ([870e2de](870e2de30f48223d8005bcf1240b624ebb314ad7))
- Add some docstrings to the functions of the translator ([6f9215e](6f9215e5fc35287517d884a702bf887d7a09aa7f))

### Feat

- Implementation of a LSTMCell class that can be translated to VHDL ([ace37fe](ace37fe4b215327bc5b43344ffcd0c44a4822dda))
- Add ability to pass kwargs to the translate function of a translatable layer ([196812e](196812eecd0dc49a1b8c2d6675b9018ca07e003e))
- Add a protocol specify a translatable layer ([0fa966e](0fa966e7f99ef2adb19321b3ca92202616b4c0a2))
- Introduce translation arguments ([2c3a8c7](2c3a8c72cfe8df70fd960e692d4fe037e2e86b6f))
- Abstract LSTM cell takes float weights instead of FixedPoint weights ([a5818cc](a5818cc0edd918ef3ca49e843738823e988bfd79))
- Add a build function to create an abstract LSTMCell object from a PyTorch LSTMCell ([baca5bb](baca5bb6c22692cf9bfc02a9147711b8869930fd))
- Use __init__ files to simplify the usage ([3cc07ee](3cc07ee048a349ef5a6a5383dcd829d64b48de2d))
- Implementation of the mapping of a torch module to the corresponding build function ([b076fa3](b076fa32cef3c64f8fcc45df24814f4333c90b5c))
- First untested draft for the pytorch translator ([7e59462](7e5946259381af397e1ccd25006815af8256026f))
- Add the ability to infer the build function from a given layer object or type ([306df14](306df1427177d15c1b1e2c59b2e774a2a6e2c471))
- Add an example using the vhdl translator for pytorch ([395adcd](395adcd3e843b7f55f6156ba183dc8800055ef51))
- Implement a more functional build function mapping ([1425e03](1425e0304cf35617106199936d3b014c0d8ca483))
- Pass an DTO to a translatable instead of raw arguments to fix typing errors ([4738725](4738725d09ca9114064c4c42dd2818fc6d5c973b))
- Add LSTMCellTranslationArguments to __init__.py file ([061ead4](061ead404dc82ddc79ac75c155328ad5733eb04a))
- Change build function mapping to a different approach ([b1b79b2](b1b79b2e5e9ea0cf627b16a41f1f75bf434b795e))
- Make build function mapping more general so that it can be reused for other frameworks ([3369d7f](3369d7fb6a7d08930514a7c0553c9efe65fc54b9))
- Removed the possibility to get a build function from a type ([dbc2e8f](dbc2e8ffd95f5ddc2476fede9d170c9d4eb020c2))
- Change translation from LSTMCell to LSTM ([5e4f1cf](5e4f1cff380fabd3685660a0c279b9098c4ef278))
- Adapt the example to the changes of the translation ([6a5644e](6a5644e30a7cd00ed1be1c2cb6fa2e0b4b114c1e))

### Fix

- Fix test ([528910c](528910cf3fe28958ebb7b246104e83df77bbf3f4))
- Fix wrong pytorch lstm cell class path ([85a733c](85a733cb5ff821bb602b5021f6438b7d5909382e))
- Fix mypy typing errors ([e1dba31](e1dba317585c269ad58719184fb4764cc66485ae))
- Use LSTMTranslationArguments object instead of a dictionary ([98a4d97](98a4d97f8fbd217f67ed4009ab63ccc4705f720d))
- Rename LSTMCell translatable to LSTM ([e05cd04](e05cd042daf0420b2046607e00eeef3606a6defb))
- Remove print call ([55164b7](55164b78c61f37f4cdadde0385965ee540e4f555))

### Refactor

- Remove custom template mapping, fixed point args and use protocols instead of abc ([fed2658](fed26585c8ccd123e590476b8e0a8ec4df8891f6))
- Change typings and Translatable yield VHDLComponent ([eedacb1](eedacb16afaf805eb6a990aa1ad40273722e02a3))
- Use better typing ([86a019d](86a019d6d3db8696850b65047481e9566da66cd8))
- Correct name of a test ([73d360f](73d360f9f3c9fc6fdf5380ff45c947b49f475199))
- Vhdl module type is now an iterable instead of an iterator to be more flexible ([1b471ca](1b471ca3a8f5b3ff3c7c28e105ae3f7f2419367d))
- Change some names to make the code more understandable ([a8d8b0c](a8d8b0c2fd3a27911a530db20dc3596113fc80e8))
- Change the name of an example ([606c0a3](606c0a30e37e5bd7d7ddc5529c770594debd7605))
- Remove empty module ([03edaca](03edaca097759ff381b012f757631662c4b5fe3a))

### Test

- Add tests for the abstract LSTMCell layer ([643a91f](643a91fa8f569fb002200039a253c9a9a79e5373))
- Add tests for the translator and the lstm_cell build function ([a92987d](a92987dfff0e3e2ab7646e984d5309383a0f9681))
- Add test that should pass in the future to check the correct layer ordering ([3e5b452](3e5b45266454ce8df5858990314e8702a0db0345))
- Add tests for the build function mapping ([4e4fee2](4e4fee2971a4131174dc6286ff85d9f4e0795611))
- Fixed tests of the build function mapping ([7885cdb](7885cdb30a413070816442c0e6daf2c0400b2743))

## 0.10.1 - 2022-06-29

[aaf7ff9](aaf7ff97e6661209388845c25686f2f2e88f702a)...[0288e04](0288e04e546dbc91d60884588fe5a31e0f81fa7f)

### Chore

- Try to fix/investigate the error with the semantic release tool ([7697877](7697877c44fa382bf0fd3838077078d61b5117dc))

### Fix

- Try to fix the error with the semantic release tool ([bc115f8](bc115f899bd85e720448bfa67fe9964bb56c594b))
- Fix error in the main.yml ([4a6ff5e](4a6ff5e61f35661a3ef83ce4335c109333834d6d))

## 0.10.0 - 2022-06-26

[de00f90](de00f90e6d4c9a17ec5831ad73b371aa7a9ac822)...[aaf7ff9](aaf7ff97e6661209388845c25686f2f2e88f702a)

### Chore

- Update numpy, onnx and add pre-commit to dev dependencies ([a23c00a](a23c00ad3faef0ed5e2318f83553ad243749c920))
- Add matplotlib dependency for the qlstm example ([dadbc20](dadbc20f5e4328d6475c418277b08059b9ba1391))
- Removing the compilation steps for onnx as they are no longer needed ([1118ee4](1118ee4ad89713a56d8a36fb93be46f0a2a33a32))

### Ci

- Use python3.10 for semantic release ([d7c5b6b](d7c5b6b6fc59ca88532defeb48894a0c792601d6))

### Docs

- Add a docstring with an example to the FixedPoint class ([961d766](961d76678d366730f57bbd69b43c38124c003bf7))
- Remove compile instructions for onnx as they are no longer needed ([3bee70a](3bee70abe4185a0a6708ffc998bdc74004f90b8a))

### Feat

- Format_vhdl function blocks the process until formatting is complete ([a8a1bd0](a8a1bd0e7a4db075d0cef4a9eb125a860a697719))

### Refactor

- Remove unused file ([577c91e](577c91ed4279ed7dbcdae71b5f4e8f868f6092ab))
- Apply python3.10 typing, renaming functions/classes, remove unused imports ([15f4b8a](15f4b8a52c78a680e3ad95fc70dbd85864282606))
- Apply python3.10 typing, remove unused imports ([f2a31c6](f2a31c6d7d75e1f545ea63cd1ed6f19dc7be7249))
- Move _int_to_bin_str to ToLogicEncoder class and refactor the class ([c6495a0](c6495a05c77962ce4cfb4a4110bf0add74d11869))
- Set correct typings ([600f6fb](600f6fb9db4e908e7c6eda4652af858258c903aa))
- Add missing typing ([1e58596](1e58596b12eef51de75fe01f60529271f4caaa6b))

## 0.9.0 - 2022-06-22

[0e62c95](0e62c950b6f56592db4a61fe2af7aae9b649d4e3)...[de00f90](de00f90e6d4c9a17ec5831ad73b371aa7a9ac822)

### Feat

- Separate hex/bin representation from vhdl hex/bin representation ([eb8fe60](eb8fe60300ee7572500f9f9d11b62a9c5abff802))
- Add a function to infer total and frac bits from a sequence of FixedPoint values ([9cc2b72](9cc2b721b147628b2abf524129eeaac8f68520d5))
- Change Rom that it uses the FixedPoint datatype ([876cdb8](876cdb821ff0ac67ae2345c8a36e4a742cce0949))
- Add function to convert list of float values to a list of FixedPoint objects ([02b26d8](02b26d868cad2a5a5bed2350a2929cf362ccdca8))
- Add function to convert a list of ints to a list of FixedPoint objects ([abece1f](abece1fd38af607c5f5734aeacd77a1743ff3411))
- Integrate FixedPoint datatype in the LSTM test bench classes ([7cbb88a](7cbb88a7f77728776e0e976dcc68505b4162f0cc))
- Verify total bits and frac bits for multiple lists ([360d318](360d318db0076d9077ceb94f3f7904d95e2b12f6))
- Add function to convert FixedPoint to signed int representation ([03001ed](03001ed608ac934e8bbdcdfa1acb2fc7c163a89a))
- Integrate FixedPoint type ([b67a609](b67a6096023a51ff4882a8cdd03a7765884c8d93))

### Fix

- Change value so that it fits into the value range of a fixed point value ([b4e973e](b4e973ebb8a087351e07966821229f69dc345d79))
- Remove old brevitas code ([86a8104](86a8104cb6049dc016a5c8da08a7d2abc011935b))
- Correct usage of the lookup_table_generator_function according to the type hints ([9812ee8](9812ee85cd467e261af942b30493ac0e970ea5e4))

### Refactor

- Apply python 3.10 typing ([1c73b26](1c73b265bb8c935d8618f0355c50ca42d1b47168))
- Small code quality improvement by using the chain function ([6517cdd](6517cdd4090574d2be5bbbdf6ae68571d6679f05))
- Use resource_utils instead of importlib directly ([a7598b4](a7598b4779d98dc0a843f6a90d459f70c6d632f3))
- Merge gen_func_for_one_lstm_cell and gen_func_for_lstm_layer in one module ([06158a9](06158a92feca1023dd1b781da691a6965529c842))
- Remove no longer needed fixed point converter classes ([4fcf0d1](4fcf0d16cdcac082c19dd654210b5d37991f9139))

## 0.8.0 - 2022-06-08

[b9afe97](b9afe9718b1ca01aeac3f49a41c8ae6967b8047e)...[0e62c95](0e62c950b6f56592db4a61fe2af7aae9b649d4e3)

### Feat

- Bump python version to 3.10 ([47f5f07](47f5f0718460a966faaa937b2c6b016720434082))
- Drop brevitas support ([103f188](103f1882c8da81cdf114f10b1b76c2ce89a07cba))
- Increase python version ([02403e6](02403e6cb7d8c9acc4357d9649fd2ae0834030a0))

### Fix

- Resolve dependency version conflicts ([32bd544](32bd544b2e74b8b57497f3fd604deb5ed86ebb42))
- Fix dependencies + onnx integration tests ([f06d0f8](f06d0f8436ca2a7ed3410aee4ad36df1cdad45c0))
- Specify exact python version in github workflow ([f3ffb18](f3ffb183e86b722cec5efb31e0937c4810542aef))
- Set correct version numbers and add protobuf dependency ([260e5fb](260e5fb31c425ad9ba2ec31f2fa292961fd28ffa))
- Fix import of Sequence type ([2c463ac](2c463acdbdae0ed7dc9fa99730f53db94deb7142))
- Set more explicit python version ([9c44093](9c44093c6cd41d05a2d178e6e113bd10f7b86016))
- Change ModuleProto to Module ([cfe418e](cfe418e41889708a53c255a8a7abcd6f1648f8f2))
- Correct deps ([7935ba1](7935ba19bcbda7e47ddbc358c12af3aa2a01df0a))
- Update poetry lock file ([9230672](92306722dabe5c4196e79a7cbbebab1e75ac3e6d))

### Style

- Beautify commit # [6772407](https://github.com/es-ude/elastic-ai.creator/commit/6772407f9929e398f7e03858e91b02c52bc8e3ec) ([ecb21e2](ecb21e271271e52d63c268b311d598fb8c86af15))

## 0.7.0 - 2022-06-05

[f07eebc](f07eebcadd5db5e2748ea6a1539bc0498cc4ed09)...[b9afe97](b9afe9718b1ca01aeac3f49a41c8ae6967b8047e)

### Feat

- Start implementing FixedPoint datatype ([8c4f420](8c4f42097ff416f8e9056af430bda01a5bd42df5))
- Add rich comparison methods, multiple operators and a bit iterator to FixedPoint ([116b006](116b00647c05ef6854d3cbd1ab0f79c58f0c450d))

## 0.6.1 - 2022-05-27

[329f779](329f779e97bcb9e175409564b733f7a756996143)...[f07eebc](f07eebcadd5db5e2748ea6a1539bc0498cc4ed09)

### Fix

- Saving generated examples to a directory instead of a giving an explicit file path ([eb41d8d](eb41d8db9af5171ac2826f41e98b5d85598b582d))

## 0.6.0 - 2022-05-25

[77adbfd](77adbfd77513f261edfa6743f2be45dd59208046)...[329f779](329f779e97bcb9e175409564b733f7a756996143)

### Fix

- Fix previously broken imports ([bf694f8](bf694f80fbd3a5478d99e8ae6b198a9e363569c9)), BREAKING CHANGE:move modules out of generator package
- Move missing files ([e4ae3c2](e4ae3c2815a33b8f4f33c9578ab5cae0842277aa))

### Refactor

- Remove usage of protected functions in tests ([47ca401](47ca401e9c19f3f80140bc9c06c1a3e162c6849c))

## 0.5.0 - 2022-05-25

[9f78ef1](9f78ef1d5de772e05a25affd7ee37788613110a4)...[77adbfd](77adbfd77513f261edfa6743f2be45dd59208046)

### Chore

- Remove deprecation warning about Q* layers ([c696596](c6965961f37a5154356a9b299fc1de36888cd184))

### Ci

- Remove outdated gitlab ci configs ([07832c8](07832c85f62e3a71bed507d100685923d70bf424))

### Docs

- Shorten ([e535ea0](e535ea0fd9d783f29ebb32d756077289d8baa8c9))
- Fix table of contents and section headers ([ecdef5d](ecdef5da63c2c10e61a159c144c5c3707a5699e8))
- Add git commit message scopes ([fe8e328](fe8e328eda5a5f9e4cac886fcbfc9388f13d3d0f))

### Feat

- Add multiline template expansion ([309ea35](309ea350fae2b4e54bf06101aadc28e227d30cbb))
- Make IOTable iterable and fix types ([faa1d77](faa1d7799bd6e8223cc4953170286d425255bb7b))

### Refactor

- Make IOTable grouping an IOTable method ([c97ec8c](c97ec8c40e1f525a19cdc6838f73be312c209b10))
- Move implementations to packages corresponding to scopes ([fa4487b](fa4487b6d491f2f3b089000aca7fe04366b441d0))
- Use correct numpy typing ([3d3ce3f](3d3ce3fe11e96c882e5392cc98ca059addb2b145))
- Move type defs ([1796473](1796473e2cfb6c0e97c9562844f811878b2b518d))

### Test

- Start implementation/integration of truth table design ([40f5396](40f5396f6a207cb72b961a2900dbbefd59dbc5f1))
- Create list from grouped tables to allow subscript ([0027a3d](0027a3d7faf88d7c2c91f325685953f5fde4e347))
- Move some test files ([dc7056c](dc7056c177dff7517d16f3adf5fbfe568eeb85f1))

## 0.4.2 - 2022-05-24

[12d377b](12d377b32d2fb4ecc8cece066a09dbba3df96cd3)...[9f78ef1](9f78ef1d5de772e05a25affd7ee37788613110a4)

### Fix

- Fix a bug and add some parameter checks ([a78e9e8](a78e9e8f669c477d0629695f5c7c8ad8628f0522))

### Test

- Add tests for the _int_to_bin_str and _int_to_hex_str functions ([002d7e2](002d7e2cc20d6646973eb343d787392b28d65b26))

## 0.4.1 - 2022-05-24

[0357cb8](0357cb84e621c99a6b01ecad9124b38456841f2f)...[12d377b](12d377b32d2fb4ecc8cece066a09dbba3df96cd3)

### Fix

- Minor errors ([812809e](812809e1d0e706df3a0514b3503dc283ea12d7a4))

### Test

- Fix errors in the intergation test of generating testbench ([d8378bf](d8378bfa6afaaf84949b284c6b53884d5b5d4ff6))

## 0.4.0 - 2022-05-23

[a7b76dd](a7b76dd4701ce32e19537d1abd2f577699203b48)...[0357cb8](0357cb84e621c99a6b01ecad9124b38456841f2f)

### Feat

- Allow ToLogicEncoder to register symbols in batches ([9209279](9209279debe651b653d2fee44533ccbdae945b32))

### Fix

- Improve names in scope of ToLogicEncoder ([67f7312](67f73129faefe343e9fb5e84563d125b1d36bab6)), BREAKING CHANGE:rename numerics attr to _symbols,
mapping attribute to _mapping.
rename add_numeric to register_symbol

## 0.3.10 - 2022-05-23

[800fc4e](800fc4ec7ac3cbd717aaadbd3652207ee51760ef)...[a7b76dd](a7b76dd4701ce32e19537d1abd2f577699203b48)

### Chore

- Add coverage and onnx outputs ([a034785](a03478528034243c1cbe8358890bb65a2845423c))
- Add htmlcov produced from coverage ([5179c0d](5179c0d526dd549fe06101342a33f89117acc022))

### Ci

- Ignore main branch for checks ([56a6640](56a6640d9839447880ca6b8e6ca495615bc86454))
- Use pypi auth token instead of testpypi ([fa2faae](fa2faae0e9fc4dc09b9de9f8b9c5032dfc104ecb))
- Remove main branch from pull_request branch-ignore filter for checks.yml ([c081fd9](c081fd984f3f8ea8bfa3bc4552980608d73badc3))
- Publish release on github ([7c0fb1c](7c0fb1cc74956257ce3fa93288c7f4dffacfef54))

### Fix

- Fix some mypy errors ([35b8fdf](35b8fdf4cb0736770d9592f86499192e1e84d673))
- Add missing mlframework types ([3b5cf5f](3b5cf5f8be829e109db363c25ecff76634f9d94f))

### Test

- Move brevitas tests to integration tests ([7ba7757](7ba7757e21245f6c418b5c12f5e3d1cc0bee9a7e))

## 0.3.9 - 2022-05-22

[3238589](323858945e70baff146ee45a76206029f3d5537f)...[800fc4e](800fc4ec7ac3cbd717aaadbd3652207ee51760ef)

### Chore

- Update version number ([94bdcab](94bdcabfa74fe4e634932eac6e4b5e36a02df236))

### Ci

- Enable test-and-publish for main ([78870c1](78870c1400b72c7d847b3da2d346f8eaf14fd619))

## 0.3.8 - 2022-05-20

[3c4c2e1](3c4c2e181384367c6af92c6d9edd30f5110ee718)...[3238589](323858945e70baff146ee45a76206029f3d5537f)

### Chore

- Try semantic release ([8a23dbf](8a23dbfeafeae82f1332c3cd28c5cbf72215a9c8))
- Setup semantic versioning ([ac06cf2](ac06cf26b4cfd66a3b49b82106ace1f236c01eb4))
- Use emojis for automatic semver ([93a60cc](93a60cc2755c08098b9c1a1f8ff5dfecee289c76))
- Remove tag_utils.py ([a8baca4](a8baca48073d6efa1330f82f87f23ec205ac02e9))
- Revert to angular style commit messages ([55f99dd](55f99ddd6f809169f91d707a51f29477523f26b0))
- Automatically update changelog ([45bfef3](45bfef38bd0dc3e86a9a291553b1f3ea5570dc9e))
- Deploy to pypi instead of testpypi ([18aee87](18aee872212ba9f066d579e4c2a5edd11e5b4a59))
- Add node_modules ([23e1234](23e12348b598edea69cf0a79e4bee26c45f62f43))
- Build via semantic-release ([e78e882](e78e882bb02b0a7adc9ff10c437a37bf6cc08dbc))
- Upload to github ([cd1a8d5](cd1a8d5a14a462db4bde32b769f88fa15aebaebc))
- Manually trigger precommit workflow ([f6611d9](f6611d9360f2b8a9ece7ace714050c85884fd6ce))
- Trigger precommit on push ([391cc8e](391cc8ef81c2d92bf432adb7814cbe95e9961c38))
- Configure hook stages ([1d9ffc5](1d9ffc57bdad9832c286d70412a2dcccce866f29))
- Correct token for  test pypi ([112eb37](112eb374c0f4b43b61b4988e8425a82881bd6802))
- Default install commit-msg stage ([28c4716](28c4716672a446623ad957b5f7f090f1eff211af))
- Remove pre-commit usage ([9cd3f34](9cd3f34c8e8b6ef1dc0904f071b1d2e3a2c0e684))
- Put mypy+dead into manual stage ([98d9620](98d9620f3a33a42a14d3dae04841660e82187609))
- Add mypy cache ([0a8c31e](0a8c31e0045ad244192bdb9fc91803a5d6470de1))
- Add npm files ([352c38f](352c38f3c83982b3abd52eb0d2bb1a654ff9bb57))

### Ci

- Add test publish workflow ([473e533](473e533dd8046e922416e121e60971a9ce2e9641))
- Update repository url to test pypi ([fab7dbb](fab7dbba3bd800189b04e1f13434440b6b1be603))
- Use pre-commit in checks ([c3b741a](c3b741a9554dd63ec02ee1044a96bfd2fa658bfe))
- Move unit/integration tests to checks.yml ([a3ffe51](a3ffe5137af975d516d00240f1517f58fbed9196))
- Disable mypy typechecking ([f63ded1](f63ded1fa71a9137d9e2502e7e9b693682116302))
- Add next version placeholder to CHANGELOG.md ([2c94335](2c94335e57dcac4cdfa1d612a4858e5fa0c34a8b))

### Docs

- Update changelog ([e1aa8c9](e1aa8c93554fc15c25a586b8e89eecda6dc03514))
- Add brief explanation of pre-commit ([3626bb0](3626bb07cc1c8600b193bc380ae8275116ebaba8))

### Fix

- Fix syntax error ([895326d](895326d67eb7ba1bb866a45c8b149778c93dc043))
- Bump version ([2cb3a72](2cb3a72b2aa9a86c0b4da71e3d7bff962a5728f6))
- Close brace ([a6e4b99](a6e4b999dadd163881fa96d03977c9c392a9267b))
- Typo unit-test -> unit-tests ([1dbd71f](1dbd71f5f3dae489b4752a5f6fdf9d10e4251a73))
- Fix typo ([7f86205](7f8620502ee544917db42ea12c7cb2eadbaef8cc))
- Fix job dependencies ([6a7d3ee](6a7d3eeb975ca303aa30fce21bd29d14cf9982d3))
- Correct numpy namespace for type alias ([a6c5842](a6c5842920c00ae6e53e226650e0fbfe48aac44a))
- Add missing import of itertools ([a6b0344](a6b0344ac4b933112b19b8603358a0adc7274533))
- Set git user+mail to gh-actions ([174ed47](174ed478b04b846912d6b0315f1143f24bc94524))
- Install latest isort fixing broken imports ([a61ef44](a61ef445672f913ec4ebc4cc8b46c2ef9099bec7))
- Add missing tags_utils again ([910c611](910c6116600b82e2c52c7d46896d92b63954d7c7))
- Updat changelog correctly ([e76a41c](e76a41cf55463cbc2a4ffa5b2b233d49695302b9))
- Fix duplicate field ([6616cab](6616cab3b0342f0b5d0b8bbdbbdf719de56d5631))
- Add missing commitlint.config.js ([2251de8](2251de83f60823d21346aedcc2b2e9aac4c27458))

### Style

- Beautify commit # [6fe04ec](https://github.com/es-ude/elastic-ai.creator/commit/6fe04eccb8dc55714b78e1a7222113c93a0b258c) ([919ac6e](919ac6ecfc5702c9a705f3da181916c2b9265366))
- Beautify commit # [1d617cd](https://github.com/es-ude/elastic-ai.creator/commit/1d617cd289068f3c6552da1bd6e9468759cb5747) ([0bb5d39](0bb5d39e73c6b4e746f1fb0308b863273d86b7f3))
- Sort imports ([de31b33](de31b335ed9ee8cf04d3823d0b9058e54df07eb9))
- Run pre-commit tools on all files ([c22eecf](c22eecf97792e104596e6575d692c6f4564e66c2))

