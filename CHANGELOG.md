# CHANGELOG


## v0.2.4 (2025-04-22)

### Bug Fixes

- Set `include_groups=False` in all the cases when using a GroupBy as per changes in pandas (fixes
  deprecation warnings, especially during testing).
  ([`e59baeb`](https://github.com/bzrudski/vectorose/commit/e59baeb5511446fd436a6659e355816318ebcfcb))

- Update stats to fit with changes to NumPy and pandas interfaces.
  ([`f42c8ef`](https://github.com/bzrudski/vectorose/commit/f42c8efdc12427717bcf17b3b75e352e7509a23d))

### Build System

- Add continuous integration for Windows.
  ([`5fbd3a7`](https://github.com/bzrudski/vectorose/commit/5fbd3a7611d5c8752f57ba12c7d5cfacde40001c))

- Fix documentation building for Windows.
  ([`80cd22b`](https://github.com/bzrudski/vectorose/commit/80cd22b1ae3cb81e2b46e53b5c78a2aa8e96c332))

- Remove version constraints on NumPy, update dependencies.
  ([`304256b`](https://github.com/bzrudski/vectorose/commit/304256bb3437758a80f85fdc9e6bc394a5200d63))

- Rename headless display setup step.
  ([`a3f630e`](https://github.com/bzrudski/vectorose/commit/a3f630ed201c57a0812c8bd0ad4f6038374afbcf))

- Switch to bash for Windows CI workflow.
  ([`d691a42`](https://github.com/bzrudski/vectorose/commit/d691a42e31f18e111eeed62b14b9fc158545745d))

- Update name for Windows CI workflow.
  ([`a38cf01`](https://github.com/bzrudski/vectorose/commit/a38cf0157d52e47d562ddad6f0986af6b354a860))

- Use pyvista headless setup action in CI/CD.
  ([`b00c5cd`](https://github.com/bzrudski/vectorose/commit/b00c5cd4021ea40909e74eb73a7d75cea97c4ed8))

### Documentation

- Add badges to README.md and to docs index.
  ([`4b26f04`](https://github.com/bzrudski/vectorose/commit/4b26f045bc15ff6dceaa26992d554fc62976af56))

- Correct the documentation to build on Windows.
  ([`3aaf841`](https://github.com/bzrudski/vectorose/commit/3aaf841cb53c5215c0fa998ac99dfa45f8d609b4))

- Fix the order of indexing for conditional histograms.
  ([`6463fd2`](https://github.com/bzrudski/vectorose/commit/6463fd29ab4bfa683912f57303de97f3fbbfe9cb))

- Remove M1 warning.
  ([`8f2c498`](https://github.com/bzrudski/vectorose/commit/8f2c4984e4760f1e4086693ad8a8ca64d3a5c3a2))

- Remove timeout for builds on Windows.
  ([`90a7279`](https://github.com/bzrudski/vectorose/commit/90a7279ad3a3b3553d99e51dce5a64312b10a4c5))

- Replace uname with system for platform checking.
  ([`1a454fd`](https://github.com/bzrudski/vectorose/commit/1a454fd0632a666e464af814a02413df219b010a))

- Revert docs configuration.
  ([`6a9a6d8`](https://github.com/bzrudski/vectorose/commit/6a9a6d81f58b6b6c25b07114757dd24262ead11d))


## v0.2.3 (2025-04-18)

### Bug Fixes

- Remove NaN vectors from 2D NumPy array files.
  ([`b067e06`](https://github.com/bzrudski/vectorose/commit/b067e06c19c4e35a85ba23a03979638dff51aa90))

### Build System

- Update codecov action version and add token.
  ([`881a22b`](https://github.com/bzrudski/vectorose/commit/881a22b32224c1025d1c3b64f4030586cc2bb900))

### Testing

- Add unit tests for vector fields with NaN values.
  ([`0408273`](https://github.com/bzrudski/vectorose/commit/0408273cafaccce9dbf172b80471513804eb7387))


## v0.2.2 (2025-04-09)

### Bug Fixes

- Correct key issue in data frame grouping for polar data.
  ([`7578a90`](https://github.com/bzrudski/vectorose/commit/7578a90d73e7d0d391042de502027cb6f94c0117))

### Build System

- Add issue tracker link to the project info.
  ([`546a600`](https://github.com/bzrudski/vectorose/commit/546a6009bdebc6da4aaadafa5ebb7d1b33208228))

- Increase timeout for myst-nb in docs.
  ([`5659cb0`](https://github.com/bzrudski/vectorose/commit/5659cb0e4c6acc8df4e0e464c7c20a796ccc9fa1))

- Remove unnecessary extra for PyVista.
  ([`f904837`](https://github.com/bzrudski/vectorose/commit/f904837cf5169a9a6572b25641c93f7acac0b51b))

### Documentation

- Add directional statistics details to repository README.md.
  ([`adba089`](https://github.com/bzrudski/vectorose/commit/adba0891188ede3dd31179795ca3b91ff86c828d))

- Add GitHub icon to online docs.
  ([`de81cbe`](https://github.com/bzrudski/vectorose/commit/de81cbee5526eb0479e209b17abea547adb2c756))

- Add missing CSV file.
  ([`724b801`](https://github.com/bzrudski/vectorose/commit/724b801a398f60c3cca8ef4b3210817732f9a50f))

- Fix bad escape sequence.
  ([`b2b6659`](https://github.com/bzrudski/vectorose/commit/b2b66594a86db1fd0c9d2205fc94405eb88ebe92))

- Replace property role with attr.
  ([`748e140`](https://github.com/bzrudski/vectorose/commit/748e14033d1649c792954f92e409f64c73e7802d))

- Switch README.md to use absolute links for PyPI compatibility.
  ([`3d691b4`](https://github.com/bzrudski/vectorose/commit/3d691b4450d42583e79fb276aaebe122f2a8a2eb))


## v0.2.1 (2025-04-02)

### Bug Fixes

- Add download role to linked files in the docs.
  ([`6072e05`](https://github.com/bzrudski/vectorose/commit/6072e05614424af15d305c725536022054b9e957))

- Deactivate xvfb in the docs pages and index.
  ([`4ecfd0f`](https://github.com/bzrudski/vectorose/commit/4ecfd0fd8e595a29ed047dd547778a2d0b90e6be))

### Build System

- Add project urls.
  ([`44a866a`](https://github.com/bzrudski/vectorose/commit/44a866a2dd57a792be2787ad24563727615502d4))

- Add PyPI release to CD job.
  ([`1d67a16`](https://github.com/bzrudski/vectorose/commit/1d67a1680207db473195e8a8952883c980d5140a))

- Manually increment version and correct reference to project version.
  ([`0f2b14d`](https://github.com/bzrudski/vectorose/commit/0f2b14d24ec9274d7d840d3a2a6ccc17adcae823))

- Use poetry for the docs dependencies.
  ([`4451bb6`](https://github.com/bzrudski/vectorose/commit/4451bb6b279faa8f292b93f22b0d2d52d9430710))

### Documentation

- Activate the XVFB in the docs index.
  ([`ed4cbb9`](https://github.com/bzrudski/vectorose/commit/ed4cbb980d3e52c531bcc62ef0d2ee07527f887d))

- Activate the XVFB in the docs pages.
  ([`c8ccfc9`](https://github.com/bzrudski/vectorose/commit/c8ccfc9ab627e6a2058b49662cfa06f673321513))

- Add xvfb to the rtd configuration.
  ([`93e05f7`](https://github.com/bzrudski/vectorose/commit/93e05f70f375160295dc9cd9664ab67d6293f615))

- Correct cross-references and data file downloads.
  ([`d51e764`](https://github.com/bzrudski/vectorose/commit/d51e764845669121cd8da95f06893a7ce846c11b))

- Delete outdated example.
  ([`af7600b`](https://github.com/bzrudski/vectorose/commit/af7600b24e92328cd9078dd8a4fb6286cb07f160))

- Remove software plugins section for now.
  ([`ba7c020`](https://github.com/bzrudski/vectorose/commit/ba7c02004db9e2426e93b8aa554b490887552744))


## v0.2.0 (2025-04-01)

### Bug Fixes

- Add ability to pass vectors in degrees to the spherical coordinate conversion.
  ([`d9a6850`](https://github.com/bzrudski/vectorose/commit/d9a6850ed929824be08223e3cf2115eacdeaa162))

- Add missing reference.
  ([`35f4df1`](https://github.com/bzrudski/vectorose/commit/35f4df1081e59dd37aba883b34d20766043e28c2))

- Add offset when magnitude range is not specified.
  ([`f2c9e4a`](https://github.com/bzrudski/vectorose/commit/f2c9e4a188ad9234468ef3a19d0daf26a515d347))

- Add padding to the polar histogram titles and change default figure size for polar histograms.
  ([`678f6aa`](https://github.com/bzrudski/vectorose/commit/678f6aac81f4c91b258fb6363d6701a35de50f5f))

- Add property to indicate if the data are axial.
  ([`6d55cd3`](https://github.com/bzrudski/vectorose/commit/6d55cd34ed260d13302f3a4f68f0a4315a125b17))

- Allow normalising vector field that contain zero vectors, switch to using np.linalg.norm for
  magnitudes.
  ([`a546b1c`](https://github.com/bzrudski/vectorose/commit/a546b1c5b8971b56439274a3cebc0b70fd1cdb19))

- Change references to biblatex.
  ([`9baf18a`](https://github.com/bzrudski/vectorose/commit/9baf18ab78ea1132e52df0de2d439c069ce8b170))

- Change scalar histogram to bars from stairs, change the polar histograms to return a figure, minor
  formatting changes.
  ([`2ca6026`](https://github.com/bzrudski/vectorose/commit/2ca6026f88ba546395fe64b3c1f3773040c51aa5))

- Change the order of updates for programmatic rotation.
  ([`99876bc`](https://github.com/bzrudski/vectorose/commit/99876bcad2a37b1e82f61cc421e70a24778d80de))

- Check number of vectors before squeezing.
  ([`1dff551`](https://github.com/bzrudski/vectorose/commit/1dff551862aee9a908b516c9d7d5427346fa2b31))

- Convert vectors to float on import.
  ([`f74df28`](https://github.com/bzrudski/vectorose/commit/f74df28b24cd9cb882e762dc753f3ff5675bc1e5))

- Correct 2D histogram export for changes in data formatting.
  ([`922b9f1`](https://github.com/bzrudski/vectorose/commit/922b9f1046c32c77a4a7f57833924d4b170e34f2))

- Correct angle calculations, fix bug that made creation of axial data in-place, fix text definition
  of angles in docstring, remove unnecessary binary search implementation.
  ([`ed42674`](https://github.com/bzrudski/vectorose/commit/ed42674ab1de4cc5ba7f3cec4b0e841912ae4f43))

- Correct off-by-one error for polar phi bin definition.
  ([`3861ac5`](https://github.com/bzrudski/vectorose/commit/3861ac5a2f84d847875b937b371398fe70755ac1))

- Correct reference formatting.
  ([`183b680`](https://github.com/bzrudski/vectorose/commit/183b6804ad1081e4a8a0b663f12f66666b177262))

- Correct the data colour limits in cases when NaN values are present.
  ([`09d95cf`](https://github.com/bzrudski/vectorose/commit/09d95cf5a06585012f095d48eff27506ef1c4202))

- Correct the ordering of the tregenza sphere faces.
  ([`3a9f7c9`](https://github.com/bzrudski/vectorose/commit/3a9f7c9f8133d1808ffaca5121b16267328d2678))

- Correct the problems that led to invalid values in arccos.
  ([`151a006`](https://github.com/bzrudski/vectorose/commit/151a006947e1819e270eb36b0c93d16170dd3738))

- Correct the sphere plotting to allow custom bounds for normalisation.
  ([`8cac41c`](https://github.com/bzrudski/vectorose/commit/8cac41c0805874a48878803701c9da5b11ccc0e6))

- Correct the spherical coordinate conversion function.
  ([`7ed1662`](https://github.com/bzrudski/vectorose/commit/7ed1662c86ad157a3312a6e26b09a6a3c6c0f7bb))

- Ensure a consistent interface in the one-bin case, remove unnecessary index reversal, streamline
  shell mesh construction.
  ([`034ef11`](https://github.com/bzrudski/vectorose/commit/034ef11dac5a0427eac932bd2c4969368566cfa9))

- Export text as text in svgs.
  ([`898277b`](https://github.com/bzrudski/vectorose/commit/898277b8d14a5a6f7c8fba9e9237efe3f9335d14))

- Fix broadcasting for normalisation.
  ([`469f242`](https://github.com/bzrudski/vectorose/commit/469f24253587dca6bd2575b1bcc40b46904597bd))

- Fix bug for passing single vectors to some util functions.
  ([`7a51f06`](https://github.com/bzrudski/vectorose/commit/7a51f06e9a5a706e5204221788c8a9e81505d877))

- Fix bug in vector normalisation with locations that would not return the locations, remove
  function to generate representative unit vectors (no longer used in statistical analysis) and
  update tests accordingly.
  ([`859e942`](https://github.com/bzrudski/vectorose/commit/859e942afdb136a41e3ded3107c4374f7d7a8756))

- Fix bugs related to stats and confidence cone plotting.
  ([`e515e84`](https://github.com/bzrudski/vectorose/commit/e515e841866596d00136d5d3e99359e274157afc))

- Fix data types to improve plotting, name the histogram series.
  ([`4616998`](https://github.com/bzrudski/vectorose/commit/46169984211cf258cbf204dd0df1de0de6f5655c))

- Fix floating-point errors leading to off-by-one errors in bin assignment.
  ([`543f592`](https://github.com/bzrudski/vectorose/commit/543f592f5765d5a045f7a74e042d217dafe18141))

- Fix magnitude weighting in triangle plot.
  ([`491e511`](https://github.com/bzrudski/vectorose/commit/491e5113d1ef3c2fca4fc6e352766fcb9fd863a7))

- Fix plotting bugs by relying on the user preference for offscreen plotting, fix triangle
  assignment in triangulated sphere matplotlib plotting, fix small documentation error in UV sphere
  construction, remove UV sphere histogram plotting function.
  ([`d48c36f`](https://github.com/bzrudski/vectorose/commit/d48c36f7b5cbf18a7ea6dfd577cb1c8cbb2cf993))

- Handle vectors at the negative pole (initial).
  ([`9665700`](https://github.com/bzrudski/vectorose/commit/96657000523489e2ba665d80534c29644c23918c))

- Improve integration of default values in plot graphics export.
  ([`cdd1dcb`](https://github.com/bzrudski/vectorose/commit/cdd1dcb6a0b53f17c3d937685b4af4a8aea11674))

- Loosen the requirement to pass a list to the SpherePlotter initialiser, add public methods for
  adjusting shell opacity.
  ([`cb5dda1`](https://github.com/bzrudski/vectorose/commit/cb5dda1524936167ca2ca9462ad333e9691d298e))

- Move the phi plot to the left in the polar plots.
  ([`00f90bd`](https://github.com/bzrudski/vectorose/commit/00f90bdcbb5f7fb8f949b503596c53ad08b21ca9))

- Remove ability to silence exceptions on vector field import.
  ([`97a05da`](https://github.com/bzrudski/vectorose/commit/97a05da88c28ccb0256178b1b6ff92cb6903f884))

- Remove deprecated triangulated spherical histogram functions.
  ([`b299ad4`](https://github.com/bzrudski/vectorose/commit/b299ad45d876dfe2a4242c060f9828e0abe09bdc))

- Remove hard-coding of the scalar bar parameters, switch to using PyVista themes.
  ([`6e30030`](https://github.com/bzrudski/vectorose/commit/6e300307217974ec1678aa061b01a4355bc6277a))

- Remove NaNs on import.
  ([`401bc55`](https://github.com/bzrudski/vectorose/commit/401bc55ce575a1ce821a70fd58330dd1d2aebf96))

- Remove shading in 3D sphere plot.
  ([`8a2e550`](https://github.com/bzrudski/vectorose/commit/8a2e55099e1ff0a5622bcf0fd5010dd2cf67834b))

- Remove some default PyVista plotting options.
  ([`a55aa00`](https://github.com/bzrudski/vectorose/commit/a55aa007f157b51164b535de178eb2213af8f525))

- Set correct ffmpeg path.
  ([`35f899d`](https://github.com/bzrudski/vectorose/commit/35f899d80a7202681d379a1c01d359bb4bbfd558))

- Suppress print statements in tregenza_sphere.py.
  ([`d1c4397`](https://github.com/bzrudski/vectorose/commit/d1c4397da83d5e17496967f8459ddd56a455efb6))

- Switch the median vector to cartesian coordinates.
  ([`be3914a`](https://github.com/bzrudski/vectorose/commit/be3914a0572d0d459a438c1972f26ab6bb4a34b1))

- Switch to more conventional angle definition for axial data.
  ([`efd406e`](https://github.com/bzrudski/vectorose/commit/efd406e20c32e416473a073b42abc8242b3eee7c))

- Try to correct median confidence cone.
  ([`7390284`](https://github.com/bzrudski/vectorose/commit/739028469966407d42410798139605a5a47ad305))

### Build System

- Add black as a dev dependency.
  ([`c650085`](https://github.com/bzrudski/vectorose/commit/c650085bff668267f45bbb9eef53bfd20010ea31))

- Add ci-cd workflow.
  ([`4454f94`](https://github.com/bzrudski/vectorose/commit/4454f9471b936d81c81159c5f6aa4e0028f35a90))

- Add dependencies necessary for animating using pyvista.
  ([`4081b3f`](https://github.com/bzrudski/vectorose/commit/4081b3fde0ca2490f32e9f6228f7f289ea8fd8e2))

- Add dependencies to use vectorose in jupyter notebooks.
  ([`1c4ba33`](https://github.com/bzrudski/vectorose/commit/1c4ba3354b6e5a71e0e4bc2865b71c5c9c6a5698))

- Add extra dependencies for documentation.
  ([`8d91580`](https://github.com/bzrudski/vectorose/commit/8d91580841e0c64ed87dc19e5e033a0955317c9f))

- Add flake8 settings.
  ([`eff9533`](https://github.com/bzrudski/vectorose/commit/eff95336d3a2b67ad12a3adc73a33d8542105287))

- Add imageio-ffmpeg as dependency.
  ([`428d2f9`](https://github.com/bzrudski/vectorose/commit/428d2f9ed2ca82a3588b70de8822ff64f87ce3ec))

- Add missing dependencies.
  ([`710cbe7`](https://github.com/bzrudski/vectorose/commit/710cbe74cd577d531b2b32ea2a5c31063ac66716))

- Add pyvista as a dependency.
  ([`905c486`](https://github.com/bzrudski/vectorose/commit/905c486f48e890e84898f6d284b8537393c9db24))

- Add sphinx-copybutton to docs.
  ([`ea3d71a`](https://github.com/bzrudski/vectorose/commit/ea3d71a00cea7e97a179a371ec7852abf38dd6e4))

- Add sphinx-gallery as a dependency.
  ([`3b92d83`](https://github.com/bzrudski/vectorose/commit/3b92d83a4ee1b785a7a03cb180d41a23a32d72fc))

- Add sphinxcontrib-bibtex to allow references in the documentation.
  ([`127a6c1`](https://github.com/bzrudski/vectorose/commit/127a6c178a539a273ead8dc649afff5fc69240f8))

- Add trimesh as dependency.
  ([`0feca69`](https://github.com/bzrudski/vectorose/commit/0feca69b971f034da2d9a29e2689f4c5bcfce640))

- Add XVFB to docs in CI-CD.
  ([`4564ed8`](https://github.com/bzrudski/vectorose/commit/4564ed8b5a20f5873e83eac6830a61153b904cef))

- Add XVFB to tests in CI-CD.
  ([`9e2dbb5`](https://github.com/bzrudski/vectorose/commit/9e2dbb59e97861ed091836ee1e13ad1c8be264ec))

- Correct dependencies on trimesh.
  ([`04087e7`](https://github.com/bzrudski/vectorose/commit/04087e785d6a33621d368818b74c966597dcfd1e))

- Disable pypi publishing for now.
  ([`14fb109`](https://github.com/bzrudski/vectorose/commit/14fb10908ff6d52fcb83056042116465ea847eb4))

- Embed python version in a string in ci-cd.
  ([`df23948`](https://github.com/bzrudski/vectorose/commit/df2394894b587ef8db234f90702a34f1cb9f0a3b))

- Fix the numpy version.
  ([`7e9000b`](https://github.com/bzrudski/vectorose/commit/7e9000b10477542ffa8668f5e135a2e4b613b0fd))

- Fix versions of importlib-metadata.
  ([`7dbdf1d`](https://github.com/bzrudski/vectorose/commit/7dbdf1debfae3255bac73f445b89ec58bce64b0d))

- Give write access on ci-cd workflow.
  ([`b3ec504`](https://github.com/bzrudski/vectorose/commit/b3ec50490c0ece6974fae78ccde47d83663f2b2a))

- Remove sphinx-gallery.
  ([`9058194`](https://github.com/bzrudski/vectorose/commit/90581947cd50b1662e4e62f6ce5e3fed1dd4d0d2))

- Set display environment variable in CI-CD.
  ([`8bca7b6`](https://github.com/bzrudski/vectorose/commit/8bca7b671822c052a188d2d85a190da575d0af21))

- Set pyvista to plot off-screen for testing in CI.
  ([`6fd5da5`](https://github.com/bzrudski/vectorose/commit/6fd5da5026bdf6378799f48f2cb3f8da40fdc730))

- Update pyproject.toml to remove deprecated fields.
  ([`3a94894`](https://github.com/bzrudski/vectorose/commit/3a94894fff7060799b8d815375bc73e3c54b066e))

- Upgrade matplotlib.
  ([`6729672`](https://github.com/bzrudski/vectorose/commit/6729672675de86bc8221252be8809255b2017c04))

### Code Style

- Add missing comma.
  ([`0ddb1f3`](https://github.com/bzrudski/vectorose/commit/0ddb1f31dbbddaeacd57adc3a9a25ff2156f97de))

- Minor reformatting using black.
  ([`7e7864a`](https://github.com/bzrudski/vectorose/commit/7e7864a731fb368894f8021de65ed81dcc1ab6c0))

### Documentation

- Add citation for the mock data.
  ([`82a47d6`](https://github.com/bzrudski/vectorose/commit/82a47d6d47695d25c6f747361452981b904e74ef))

- Add clarifications to index page following feedback and add missing newline at end.
  ([`5d48539`](https://github.com/bzrudski/vectorose/commit/5d48539f648b810e0779e4960241405998dc9941))

- Add conditional histograms to the advanced_histograms.
  ([`467450a`](https://github.com/bzrudski/vectorose/commit/467450a057bb556bd9bce682f320118c6a37d6aa))

- Add debug option for autoapi.
  ([`d2b254b`](https://github.com/bzrudski/vectorose/commit/d2b254b52e47c64f7aa83b199c3586eb825909f6))

- Add descriptions for the sphere data frame columns.
  ([`7ff487f`](https://github.com/bzrudski/vectorose/commit/7ff487f086ecdaa78e2acaad4dbfeb818ec83fcd))

- Add docstring to the top-level module file.
  ([`82a7070`](https://github.com/bzrudski/vectorose/commit/82a707033986b06c9b3d1229bae3ba89db91d7b0))

- Add example for creating and exporting animations.
  ([`9b5c21b`](https://github.com/bzrudski/vectorose/commit/9b5c21b141f8fd958f2f33100a977672429683f7))

- Add help wanted section in stats and fix clarification admonition.
  ([`5141c93`](https://github.com/bzrudski/vectorose/commit/5141c93f7654d89881627ed9a966cbe5df021aeb))

- Add installation instructions.
  ([`8354b46`](https://github.com/bzrudski/vectorose/commit/8354b46ead5a20472ad4b4fead772dd559258fa6))

- Add mac compatibility warning and jupyter instructions in installation docs.
  ([`0f45696`](https://github.com/bzrudski/vectorose/commit/0f45696e97abbf9d68c3aa41f15e575077a34cad))

- Add missing image for rotated blocks example.
  ([`ff50311`](https://github.com/bzrudski/vectorose/commit/ff50311b7257d91adc0a43a222575cccdc83512f))

- Add missing kappa.
  ([`59e771a`](https://github.com/bzrudski/vectorose/commit/59e771aed7eba83f50b90d01cb39bbf466eaeefc))

- Add missing return type to the polar histogram plotting function type annotation and docstring.
  ([`8fdb3dc`](https://github.com/bzrudski/vectorose/commit/8fdb3dc9b4466413a436e7a7f8dcb7ec3cb0f480))

- Add reference to Woodcock's original work on the shape and strength parameter.
  ([`f388760`](https://github.com/bzrudski/vectorose/commit/f388760b053184558bb87012ff810834165359b8))

- Add skeleton for sphinx-gallery.
  ([`431d908`](https://github.com/bzrudski/vectorose/commit/431d908ed00fb3fa1287a4602f7803142c950744))

- Add Statistics and Next Steps to the Quick Start guide, expand other sections in this document,
  make code executable, add quick start vectors.
  ([`cbd13ac`](https://github.com/bzrudski/vectorose/commit/cbd13ace371ea4b898d0aabf96467e7e7eeb399a))

- Add support for matplotlib animation scraping based on Sphinx-Gallery docs.
  ([`b7193f2`](https://github.com/bzrudski/vectorose/commit/b7193f2334328e630d04335a0e485011c35f21c3))

- Add the app icon and update the docs configuration, copy the user's guide from other repo.
  ([`5fd85e5`](https://github.com/bzrudski/vectorose/commit/5fd85e5334faf0abcb374bc14089540c893e46a0))

- Add Tregenza reference.
  ([`bc67555`](https://github.com/bzrudski/vectorose/commit/bc67555e56aeeaa371e8012a0e52d4ec54b8efc2))

- Add trimesh to the intersphinx settings.
  ([`d541a04`](https://github.com/bzrudski/vectorose/commit/d541a04a81a9f1ed9c0b44c41fe7adc65c7eca3d))

- Add warning to coplanarity.
  ([`e368386`](https://github.com/bzrudski/vectorose/commit/e36838622660f77eb851f0f191ec460e52773447))

- Added clarification to the docs for `generate_representative_unit_vectors` explaining the cap on
  the number of vectors.
  ([`0b9686c`](https://github.com/bzrudski/vectorose/commit/0b9686c9f5f890487d898f20667f4927ddfc2990))

- Begin page about advanced histograms, clarify a bit on basic histogram plotting.
  ([`281e0c0`](https://github.com/bzrudski/vectorose/commit/281e0c0c2e4a65d8f480f534e99fb16e42d0c0b1))

- Begin quickstart, begin rewriting statistics page.
  ([`730072a`](https://github.com/bzrudski/vectorose/commit/730072a3d52a2df13936ff539db6ae298e47f5e1))

- Change the theme for the sphinx documentation.
  ([`9ffaab7`](https://github.com/bzrudski/vectorose/commit/9ffaab71598d548bad9d3de3204c952184c7ce12))

- Changed formatting to remove false todo.
  ([`87bf81e`](https://github.com/bzrudski/vectorose/commit/87bf81e5e703b3ffcd666a6dfed72c5029c084ed))

- Clarify that SpherePlotter is for PyVista.
  ([`35dd466`](https://github.com/bzrudski/vectorose/commit/35dd46664210e9f88ca7035cef6dee55a8d4e99d))

- Clean documentation in triplot.py.
  ([`915a333`](https://github.com/bzrudski/vectorose/commit/915a33315f23735c2cced3315d07553093a791f5))

- Content and formatting corrections following feedback.
  ([`e28c155`](https://github.com/bzrudski/vectorose/commit/e28c155bc1fd91ce1bd3ca0fdb21c46c3d3bd9a0))

- Convert the data_format page to notebook-based format to allow the code to execute.
  ([`1853f77`](https://github.com/bzrudski/vectorose/commit/1853f77c8ce380178f17ec2b3139a2d42bdec507))

- Edit exclude patterns for sphinx.
  ([`7b6574d`](https://github.com/bzrudski/vectorose/commit/7b6574d3f50c57f48c401d1513eb69fb9cd919ed))

- Expand users' guide, add references, update docs configuration.
  ([`7bf14a6`](https://github.com/bzrudski/vectorose/commit/7bf14a690f53f24e33d7bcc99fc2d25b7294fd05))

- Finish draft of the statistics manual page.
  ([`938d5b6`](https://github.com/bzrudski/vectorose/commit/938d5b639e78af640753282e15389a2184c5a04c))

- Fix citations for the statistics.
  ([`b6207bd`](https://github.com/bzrudski/vectorose/commit/b6207bd71590f0c6f618fac3211eb25aad1f2f11))

- Fix formatting for the AngularIndex enum docstring.
  ([`1d29979`](https://github.com/bzrudski/vectorose/commit/1d299799d7abcc261982acf2c9d98ecba9effb92))

- Fix typo in reference to Reznikov et al. (2022) in the introduction to vectors.
  ([`677d0ba`](https://github.com/bzrudski/vectorose/commit/677d0ba3b99669564c83a9e14510ce68d334bab2))

- For now, remove the link to the plugin repo (to add back later).
  ([`d7b67ac`](https://github.com/bzrudski/vectorose/commit/d7b67ac774ada238f7866ca4bd551e6d86304db3))

- Improve consistency in docstrings.
  ([`560b6de`](https://github.com/bzrudski/vectorose/commit/560b6def801117712e1cf702b58bced88d7b277c))

- Improve Tregenza sphere documentation.
  ([`96f793f`](https://github.com/bzrudski/vectorose/commit/96f793f0d3a5ae3c7012ae6f522457ecd89cf21c))

- Include only references cited.
  ([`0a96679`](https://github.com/bzrudski/vectorose/commit/0a9667905442784a27c42e40fbb533b982679138))

- Make plotting docs more consistent.
  ([`bbd1810`](https://github.com/bzrudski/vectorose/commit/bbd181002d017c9a2ffe9a111cbd818a710e5d02))

- Minor change in types in docs.
  ([`6276286`](https://github.com/bzrudski/vectorose/commit/6276286dae1e32047f5a709b9c451b05d9a9e295))

- Minor clarifications in the title, and converting remaining rST formatting to markdown.
  ([`6245eae`](https://github.com/bzrudski/vectorose/commit/6245eae0d3a910f3458cbb6528347a672b02b06a))

- Minor corrections to the stats module description.
  ([`fedef94`](https://github.com/bzrudski/vectorose/commit/fedef94f972d6f80e438f873acd31625ac0f61dc))

- Minor fixes for inter-sphinx documentation linking.
  ([`eda4fa8`](https://github.com/bzrudski/vectorose/commit/eda4fa85d6b5a19569c6bb18fb766c3761eaf8ff))

- Minor rename of np to numpy in docs.
  ([`9176dbd`](https://github.com/bzrudski/vectorose/commit/9176dbd631a0bc1ad19d51bfc4a2c80aa82e9f4f))

- Move the gallery to be before the API docs.
  ([`7f54431`](https://github.com/bzrudski/vectorose/commit/7f54431d3519e0db58c415c066b40b49551cb4f3))

- Remove call to x-frame buffer in examples, add code to ensure image export directories exist.
  ([`395fba5`](https://github.com/bzrudski/vectorose/commit/395fba5fdd0c064495407985aa6d3aa63feaef03))

- Remove extraneous example and remaining sphinx-gallery configuration, update copyright year.
  ([`f82b097`](https://github.com/bzrudski/vectorose/commit/f82b0977ccab17aba121fe1f071ea98a4705eb4d))

- Remove readme for sample plots (link to relevant page added in box).
  ([`2be154b`](https://github.com/bzrudski/vectorose/commit/2be154b3a97b3510dbbc5c548e92c1885c02de35))

- Remove reference to example gallery in user guide page, add help wanted sign.
  ([`34aedba`](https://github.com/bzrudski/vectorose/commit/34aedba4fd994a5e63612c81f0041eac2a1208f2))

- Remove reference to GUI.
  ([`1b6112b`](https://github.com/bzrudski/vectorose/commit/1b6112b49d13a4a59b4be1bb3653b1cf030b6420))

- Remove unnecessary keywords field.
  ([`1ae2164`](https://github.com/bzrudski/vectorose/commit/1ae21645a0a607d8ca1e200a6a99f8b5a731141e))

- Remove unnecessary todo.
  ([`33d2e12`](https://github.com/bzrudski/vectorose/commit/33d2e12cde50258ac83bdd00a922cee30ed7f3cb))

- Rename Gallery to Examples.
  ([`fa123fb`](https://github.com/bzrudski/vectorose/commit/fa123fbd180d8a615e04d21399c8c29f1f096ddf))

- Update animation example to hide slider bars.
  ([`1a90b3d`](https://github.com/bzrudski/vectorose/commit/1a90b3dbe60c3a5408b645242fb83b65161aa11e))

- Update configuration for Read the Docs.
  ([`a13782b`](https://github.com/bzrudski/vectorose/commit/a13782b11607df17a701c6b5b91306d4b25986f0))

- Update documentation index to contain a slightly modified copy of the top-level README.
  ([`0bdd619`](https://github.com/bzrudski/vectorose/commit/0bdd619568b34e4d965a690851e4afd239dee285))

- Update README.md.
  ([`5ecb454`](https://github.com/bzrudski/vectorose/commit/5ecb4544998d5ac8bb0213c1af49ffdc88414d00))

- Update sphinx-gallery configuration to improve linking to VectoRose objects.
  ([`a7a919f`](https://github.com/bzrudski/vectorose/commit/a7a919f174542d6a8390651b0995689862026f12))

- Update Users Guide, move sphinx-gallery examples to pages in Users Guide.
  ([`e1583d7`](https://github.com/bzrudski/vectorose/commit/e1583d78ea9616eb8e4a7d9a0faf0e45d3becfe2))

### Features

- Add 2D planar phi-theta histogram, small fixes.
  ([`892a3b4`](https://github.com/bzrudski/vectorose/commit/892a3b4fca8a5d8183a4486d6a70226226484b65))

- Add ability to construct 1D log-scaled histograms.
  ([`3094f47`](https://github.com/bzrudski/vectorose/commit/3094f4709f1dc040bab0dd8e6e8a10b8b9533368))

- Add ability to define confidence cone radius in degrees.
  ([`596aff5`](https://github.com/bzrudski/vectorose/commit/596aff51149f113d34cc01f4035e252895d90497))

- Add ability to export SVG and PDF graphics.
  ([`abf1189`](https://github.com/bzrudski/vectorose/commit/abf11890f59aa3ea1dc1959950e27bd1dcf5f0be))

- Add ability to fail with exceptions for vector field import.
  ([`e41060b`](https://github.com/bzrudski/vectorose/commit/e41060b9accf8796845805de841099e6425f73a0))

- Add ability to generate mock data from multiple vmf distributions, add new way of ensure that all
  arguments have the same length for these multiple distributions.
  ([`4cd9903`](https://github.com/bzrudski/vectorose/commit/4cd9903360c692e698fb140eded463d9a831f1aa))

- Add ability to get underlying DataFrame from triangulated and Tregenza spheres.
  ([`a01706f`](https://github.com/bzrudski/vectorose/commit/a01706f5103d5cab4853ec1074f95d458012488d))

- Add ability to plot specific shells.
  ([`f9c6ae9`](https://github.com/bzrudski/vectorose/commit/f9c6ae971df834648a25bf0a87e1a61fbb873062))

- Add ability to programmatically reorient the sphere plotter.
  ([`51fa364`](https://github.com/bzrudski/vectorose/commit/51fa364c4287e3e98aa7f33f6bdf7b1abc92a925))

- Add ability to show and hide sliders and legend, export screenshots, produce single shell plots
  from the same function as multi-shell plots.
  ([`57a3456`](https://github.com/bzrudski/vectorose/commit/57a345628e9afdb140ad8921131c18c0eb9935e0))

- Add ability to specify colour bar max and min for spherical plots.
  ([`b366d80`](https://github.com/bzrudski/vectorose/commit/b366d80b20b26f3b3845e329919a1e9f732d6c85))

- Add axis plotting to mesh-based histogram (buggy).
  ([`050f065`](https://github.com/bzrudski/vectorose/commit/050f065ad0c3b71e26736a73f55ff887543476f7))

- Add co-planarity, VMF fitting and new data structure for hypothesis testing results.
  ([`66a1adf`](https://github.com/bzrudski/vectorose/commit/66a1adffccce37a5397bfc5ba9fd991baf7a52a7))

- Add coarse tregenza sphere (for illustration).
  ([`306b272`](https://github.com/bzrudski/vectorose/commit/306b272ca3c0eae45d809b911928db9475c5f0f4))

- Add code to compute the mean unit vector for VMF.
  ([`30c8333`](https://github.com/bzrudski/vectorose/commit/30c8333911741d81dda7f5f86c51434c163c8457))

- Add confidence cone plotting.
  ([`4a3eeec`](https://github.com/bzrudski/vectorose/commit/4a3eeec55c0bae60c9ede9b8adea98d63bc9a845))

- Add extra parameters for importing vector fields, allowing for non-2D arrays.
  ([`56a50b9`](https://github.com/bzrudski/vectorose/commit/56a50b9150a6ff440adc9d04b2c4141bb8e533f7))

- Add FL&E implementation of sampling from the Watson distribution.
  ([`1900252`](https://github.com/bzrudski/vectorose/commit/1900252ce905ad7c60e745acfaaec2af94720e13))

- Add function to activate specific shells in plot, improve memory.
  ([`3d2cbc5`](https://github.com/bzrudski/vectorose/commit/3d2cbc5a7d336b88c49452d198c7a1a947f423c4))

- Add function to construct mesh of uv sphere.
  ([`46263b2`](https://github.com/bzrudski/vectorose/commit/46263b2e548fd12d465ec64368a87c47dfb5c207))

- Add function to convert to the standard spherical coordinate convention, fix issues with computing
  angles and plotting confidence cones, add parameters to pass pre-computed intermediates for stats.
  ([`fd27225`](https://github.com/bzrudski/vectorose/commit/fd27225c8f3e82812f2cad84ce4f952ec2d56397))

- Add function to convert vectors to a DataFrame.
  ([`55378a6`](https://github.com/bzrudski/vectorose/commit/55378a683a117341c914734bc0c9d15e48139122))

- Add function to explicitly compute eigen-decomposition of the orientation matrix.
  ([`2953578`](https://github.com/bzrudski/vectorose/commit/29535782c1a40d361ef055e3e9d263036bdf3163))

- Add function to generate vectors from multiple Watson distributions.
  ([`4f7949b`](https://github.com/bzrudski/vectorose/commit/4f7949b1a24f7fb6c21e7ac6f2f1b25354e6c082))

- Add function to normalise vectors.
  ([`cc19c79`](https://github.com/bzrudski/vectorose/commit/cc19c7979bea13f976c1fd2dd3deee9751a28344))

- Add functions to convert vectorial data into axial data and plot on the triangle sphere.
  ([`31cf67a`](https://github.com/bzrudski/vectorose/commit/31cf67ace2ea5de3586bcb49acdb021d9a8010eb))

- Add functions to flatten vector fields and to rotate vectors based on FL&E.
  ([`d2f5d22`](https://github.com/bzrudski/vectorose/commit/d2f5d220d213ac84bc8f2ada07c0a7b4ecb78cdd))

- Add method to extract vectors to compute statistics.
  ([`770d484`](https://github.com/bzrudski/vectorose/commit/770d484a33dad602615ca9ea71697f5e165779e1))

- Add new function for computing angular distances (arc lengths), including unit test.
  ([`97dceee`](https://github.com/bzrudski/vectorose/commit/97dceee592c656bb08dac0b6a3dd3d2d4bc2c6f9))

- Add new normalisation approach to create unit vectors with directions weighted by original
  magnitudes, remove default naive normalisation from stats routines.
  ([`34cce46`](https://github.com/bzrudski/vectorose/commit/34cce46af83fdc4e423c0bb3b1cc56d2e2cf02ee))

- Add plotting functions for Tregenza sphere.
  ([`8eef97c`](https://github.com/bzrudski/vectorose/commit/8eef97c6506cc56543b16be54b5ec3231e4d87d9))

- Add seeds to random number generation in mock_data.py.
  ([`25f934b`](https://github.com/bzrudski/vectorose/commit/25f934bd8a406c997b15ebb692127f1319454818))

- Add shell label option for exported videos.
  ([`13f6e7d`](https://github.com/bzrudski/vectorose/commit/13f6e7d34052dcf4e8aad2b6e570471ba4c326e4))

- Add spherical axis labelling.
  ([`8ef6013`](https://github.com/bzrudski/vectorose/commit/8ef601376513ecb9f7850a08ebc048151784d91c))

- Add statistics from FL&E for vector fields.
  ([`ce5037e`](https://github.com/bzrudski/vectorose/commit/ce5037e9ec2cecd8c054cdd1b0c9d6ca5421d68e))

- Add support for computing marginal and conditional histograms.
  ([`efa4b55`](https://github.com/bzrudski/vectorose/commit/efa4b55bc04dc464ecb9d9567d0cb1cdca885826))

- Add support for constructing meshes for each sphere type, update the sphere interface to allow
  plotting, begin implementing 3D nested sphere plotting using pyvista.
  ([`dcee8fa`](https://github.com/bzrudski/vectorose/commit/dcee8fa695e37e92121f4b6febc244a6f8dbac7e))

- Add support for exporting animations as GIF.
  ([`0e70fb3`](https://github.com/bzrudski/vectorose/commit/0e70fb30f2c233e9f406316972cbe8936fcac15f))

- Add support for other normalisers.
  ([`9e16537`](https://github.com/bzrudski/vectorose/commit/9e165379f4ae7d8c0599ee956c28a438dc0843e9))

- Add support for preserving vector locations in the polar analysis.
  ([`fb8386b`](https://github.com/bzrudski/vectorose/commit/fb8386b9fa5958e8f3e0ceb0cc4bfece518086cb))

- Add the ability to construct and plot a histogram on a triangle mesh icosphere.
  ([`6b7012f`](https://github.com/bzrudski/vectorose/commit/6b7012f8632f1971458a14b0380bdce716f7c1c2))

- Add the ability to keep spatial coordinates attached to vectors.
  ([`a463c13`](https://github.com/bzrudski/vectorose/commit/a463c13198b55022403edf58e84367d603f8f088))

- Add the ability to normalise the histogram meshes by shell maximum.
  ([`6a0b4a6`](https://github.com/bzrudski/vectorose/commit/6a0b4a6dc79109b97c97f5ee0fed251e258d4afc))

- Add the ability to return the histogram containing frequency as a proportion instead of a count.
  ([`37f62c3`](https://github.com/bzrudski/vectorose/commit/37f62c3f02e4243f28a9b1b66ebb1baaea17d62d))

- Add the ability to weight mesh-based histograms by magnitude.
  ([`a6415c7`](https://github.com/bzrudski/vectorose/commit/a6415c71f7eb8d76835772eb66b22d7c9a20d0bc))

- Add Tregenza detail level enum.
  ([`1f4686b`](https://github.com/bzrudski/vectorose/commit/1f4686b2806f74503f436252df7ae7fba7acd668))

- Add tregenza sphere support.
  ([`8965c27`](https://github.com/bzrudski/vectorose/commit/8965c27bd6483c6a2b52d17643b8f97a745dead8))

- Add ultrafine Tregenza sphere and area weighting correction.
  ([`11e38d5`](https://github.com/bzrudski/vectorose/commit/11e38d5a577a7f0a48743682d9750c5003542c66))

- Add von Mises-Fisher-based artificial data generation.
  ([`67a3f87`](https://github.com/bzrudski/vectorose/commit/67a3f870abcfd7ccd54d39c198b9a228068946fc))

- Allow all mesh shells to have the same size when plotting.
  ([`43fe37c`](https://github.com/bzrudski/vectorose/commit/43fe37cc2f23b6805285d204c5c544107431df2a))

- Begin implementing the true bivariate histogram.
  ([`1cf1a4b`](https://github.com/bzrudski/vectorose/commit/1cf1a4b9588736a40ce3a8528b8f99329203f1eb))

- Change the polar histograms to be based on pandas, remove confusing planar 2D histogram, change
  plotting capabilities to rely on the pandas data.
  ([`3ad8095`](https://github.com/bzrudski/vectorose/commit/3ad809566fa5c0d01856380291b0ad10243be7c8))

- Complete refactor for plot labelling, add ability to use log scale.
  ([`00e98e7`](https://github.com/bzrudski/vectorose/commit/00e98e70313fcaf7e328e7289c57411478835726))

- Compute the shape and strength parameters based on the orientation matrix eigenvalues.
  ([`41d0820`](https://github.com/bzrudski/vectorose/commit/41d082063c0a90174f1875046d731e000f4535e6))

- Finish fine Tregenza sphere, build full sphere, include area-weight correction and magnitude
  weighting.
  ([`e05c18c`](https://github.com/bzrudski/vectorose/commit/e05c18c21deea6bb098b2745460c76dc5feb1ba4))

- Implement the correlation coefficient for finding the relation between the magnitude and the
  orientation.
  ([`ccd0c10`](https://github.com/bzrudski/vectorose/commit/ccd0c10ade1cd7e0708f877ea0a3c3f98bcbfee8))

- Improve the ability to programmatically set the view.
  ([`1cadceb`](https://github.com/bzrudski/vectorose/commit/1cadceb81f72665211b80062e507131578ec11bc))

- Improve the interface for the SpherePlotter by adding some properties for the active shell and
  opacities, as well as visibility of UI elements (e.g., sliders) and add warning for camera angles.
  ([`72dae4e`](https://github.com/bzrudski/vectorose/commit/72dae4e4a09eae1f6ae9e510570a397dddf674ad))

- Include implementation of nested shell-based triangulated sphere histogram.
  ([`c975661`](https://github.com/bzrudski/vectorose/commit/c975661e9b59461a9eecf9949a055ea592c271fa))

- Modify protected method to plot any data on a single shell and make it public.
  ([`5650604`](https://github.com/bzrudski/vectorose/commit/5650604f14b6db01927b4efddcef7cd23bffeae0))

- Replace roll, azimuth and elevation with a function to view a specific phi and theta on the
  sphere, add new functions to expose the movie creation to allow increased flexibility, add
  properties to allow access to information about these aspects of the plotter, add property to
  check if plot produced.
  ([`2b803dd`](https://github.com/bzrudski/vectorose/commit/2b803ddfc16335ed4782013c168b1cde4e92355e))

- Replace the vector rotation with scipy-based rotation.
  ([`66b4ca7`](https://github.com/bzrudski/vectorose/commit/66b4ca7e14a6c18ee53e677e52c73f126bcd52af))

- Restrict phi polar plot to the appropriate subset of angles instead of plotting the full circle.
  ([`59f3683`](https://github.com/bzrudski/vectorose/commit/59f3683cb90e4e31231c06325e33ee8409f764ae))

### Performance Improvements

- Change the bin finding for Tregenza spheres to be vectorised.
  ([`46a5647`](https://github.com/bzrudski/vectorose/commit/46a5647388b10b83610a866e4e0204184f1599f3))

### Refactoring

- Add SphereBase as a top-level import.
  ([`61a0cfc`](https://github.com/bzrudski/vectorose/commit/61a0cfc583780e0b5b489cd07094759217c4afa1))

- Add Tregenza to top-level init.
  ([`73e2320`](https://github.com/bzrudski/vectorose/commit/73e2320ee7b7cb09e9ea88371baee1464cb08601))

- Add util to top-level __init__.py.
  ([`0336204`](https://github.com/bzrudski/vectorose/commit/0336204db05bfb266ccc804721c0bed1bab753da))

- Change the binning so that the code works with the half-number of bins throughout, except when
  plotting.
  ([`4887d6d`](https://github.com/bzrudski/vectorose/commit/4887d6de63450cc9738907b1c0ff523e6582f677))

- Create a superclass for spherical histograms to allow easier future integration of other histogram
  implementations.
  ([`ea78033`](https://github.com/bzrudski/vectorose/commit/ea78033ff1d10ebc6e9841715431936285125412))

- Fix typos and clarify a docstring.
  ([`ce8351a`](https://github.com/bzrudski/vectorose/commit/ce8351a57e7c9cc00cbe105bff2da6d9514c7137))

- Import stats in the pacakge root.
  ([`2df7823`](https://github.com/bzrudski/vectorose/commit/2df78237dc1d81dae0020ee17e0a3f299b407f08))

- Minor reformatting.
  ([`0e18470`](https://github.com/bzrudski/vectorose/commit/0e18470b91f165ddccceca922cfe3a4dd12f7bd1))

- Move histogram operations out of the plotting module.
  ([`c6f70ba`](https://github.com/bzrudski/vectorose/commit/c6f70ba3f35c7e863145f2cce7e99b178d25af92))

- Move Tregenza sphere matplotlib plotting to the plotting module.
  ([`39dca98`](https://github.com/bzrudski/vectorose/commit/39dca98340538e114307371b411b8a17646236d2))

- Remove old UV sphere-based histogram plotting code to avoid confusion.
  ([`400b983`](https://github.com/bzrudski/vectorose/commit/400b983c76553ce39be8dd7a765194c1f481c1b2))

- Remove print statement from binary search.
  ([`af97e0a`](https://github.com/bzrudski/vectorose/commit/af97e0ae54e11be1daf4125971b5085abfc7e515))

- Remove print statements and clean new lines.
  ([`3575187`](https://github.com/bzrudski/vectorose/commit/35751873831dd37ebac7ce5d857d61c2ee198524))

- Remove the mpl plotting function for Tregenza spheres.
  ([`9dc1779`](https://github.com/bzrudski/vectorose/commit/9dc177919a7b1c0563f969c96b296b7e3adf9e52))

- Remove the naive mock data functions that only worked on the 2D plane in favour of true spherical
  approaches.
  ([`3a341e1`](https://github.com/bzrudski/vectorose/commit/3a341e13d521e5e4e86a11c4f66cbb49e046f913))

- Remove the old Tregenza sphere implementation, add mpl plotting support to the new version, rename
  the new versions.
  ([`5aaf9f5`](https://github.com/bzrudski/vectorose/commit/5aaf9f5e4b54fa98efc82e8a68fd68b9b603e790))

- Remove unnecessary import, reorder imports.
  ([`265113f`](https://github.com/bzrudski/vectorose/commit/265113f3fc84dbe424675df7fb212058280bc1d2))

- Remove unnecessary options to convert data to axial in statistics functions.
  ([`a5dfbb6`](https://github.com/bzrudski/vectorose/commit/a5dfbb6e245848c8bd11d8dd784a9d721606ce38))

- Remove unnecessary variable.
  ([`2b0558e`](https://github.com/bzrudski/vectorose/commit/2b0558e9733537726f512ccef911614bbd37232d))

- Remove unused imports.
  ([`e46dd63`](https://github.com/bzrudski/vectorose/commit/e46dd630dab38d016c0257b8af0dbdc90c49fae1))

- Remove unused main script.
  ([`ce61b9c`](https://github.com/bzrudski/vectorose/commit/ce61b9cd82c40c1b13c30d9407260e7c6d7de404))

- Rename key modules and remove unused functions, move magnitude to util.
  ([`f3e95ef`](https://github.com/bzrudski/vectorose/commit/f3e95ef65b8f184db39e6486c1682e28d0275420))

- Rename the modules for triangle-based spheres and Tregenza spheres.
  ([`b109952`](https://github.com/bzrudski/vectorose/commit/b109952304caf2553b511a5fd9a2097b25e08fc1))

- Rename TregenzaSphereBase to TregenzaSphere, as this class is not abstract.
  ([`89d969d`](https://github.com/bzrudski/vectorose/commit/89d969d2f92760964e76b351bfa994bfb183282f))

- Reorder plotting functions.
  ([`4766e32`](https://github.com/bzrudski/vectorose/commit/4766e32ac5e0b86bd4dd433e1884adc40c97d216))

- Separate out utility functions into new module.
  ([`e330dbf`](https://github.com/bzrudski/vectorose/commit/e330dbf0aceb8b1bbf406a1abd669d40eb5b7a7c))

- Start removing duplicate spherical plotting code.
  ([`5266e39`](https://github.com/bzrudski/vectorose/commit/5266e3955cea52d1b155b00980a4c5f35fdfbaf3))

### Testing

- Add new tests for scalar and polar histogram plotting.
  ([`88edf15`](https://github.com/bzrudski/vectorose/commit/88edf1587facb0e11bd39ff3c22bb7ce5aaa1eb8))

- Add tests for DataFrame conversion.
  ([`5c60dad`](https://github.com/bzrudski/vectorose/commit/5c60dad99abe6e9bd3e7249499ab4200bb2d157d))

- Add tests for spherical axis plotting, SpherePlotter properties and UI element visibility.
  ([`d6fbbf6`](https://github.com/bzrudski/vectorose/commit/d6fbbf6330c3cbdf32c359cb2a93432adb4c5f55))

- Add tests for the new plotting features from previous commit.
  ([`46d6631`](https://github.com/bzrudski/vectorose/commit/46d66315e412c3502dbbd70be0d6758762f83e24))

- Add tests for the updated triangulated sphere implementation.
  ([`f21259e`](https://github.com/bzrudski/vectorose/commit/f21259e426bc2fa814367a923eea6b44e1df1f29))

- Add unit tests for preserving spatial locations in labelled vectors, reformat tests to use
  fixtures.
  ([`3cc9610`](https://github.com/bzrudski/vectorose/commit/3cc961079c380d1f078fc28b08bcfee89cc4f11d))

- Add unit tests for remaining utility functions and reformat code.
  ([`ee9a511`](https://github.com/bzrudski/vectorose/commit/ee9a51113181fc56d603a46512e135994109e3e2))

- Add unit tests for the polar_data module.
  ([`0bb6dec`](https://github.com/bzrudski/vectorose/commit/0bb6deca2470083121b36da3f8b9ff9cdc59eeb5))

- Add unit tests for the statistics routines.
  ([`944d005`](https://github.com/bzrudski/vectorose/commit/944d0058542a9b6bdab8efeede4c45ab69e0ed2a))

- Begin adding unit tests for utilities and triangle-based and Tregenza spheres.
  ([`55fe10b`](https://github.com/bzrudski/vectorose/commit/55fe10b571b0eb26c3a0392059b503da7ca8fa4c))

- Begin unit tests for plotting module. Tests currently included for some of the SpherePlotter
  functionality and most of the matplotlib plotting.
  ([`e47d224`](https://github.com/bzrudski/vectorose/commit/e47d224dc476a713d1e6ee07b7a6dd13b09c9a0b))

- Correct attribute name in mesh test.
  ([`5e8c55b`](https://github.com/bzrudski/vectorose/commit/5e8c55b5bf6f27ae39a5493af7820bcc6e34946f))

- Revamp the Tregenza sphere tests to fit the new implementation.
  ([`e4342c8`](https://github.com/bzrudski/vectorose/commit/e4342c878c7fb5a0e8db31b469e9b2e767dd74a9))

- Test colour bar plotting in 3D.
  ([`d976b19`](https://github.com/bzrudski/vectorose/commit/d976b19d6b6757b9e7a61369cd3b910b581c9c24))

- Write vector import tests.
  ([`3843502`](https://github.com/bzrudski/vectorose/commit/38435024c7848f15c288da054fde27a3e84317a1))


## v0.1.0 (2024-01-17)

### Bug Fixes

- Add version information to the package init.
  ([`7687297`](https://github.com/bzrudski/vectorose/commit/76872972953e0f923f43ef07a0a5ffac631e6107))

- Add vscode folder to gitignore.
  ([`fb1269f`](https://github.com/bzrudski/vectorose/commit/fb1269f589537558812772ee2438c50b614603cf))

- Change type annotations to be python 3.9-compatible.
  ([`11b34a7`](https://github.com/bzrudski/vectorose/commit/11b34a7bfa4f4a32f2d2ac0ca49d0a77c1fb7e39))

- Correct matplotlib imports.
  ([`3f673b9`](https://github.com/bzrudski/vectorose/commit/3f673b9a9eafa3559bdbd1a45aba10686430ebbd))

- Correct more types for python3.9.
  ([`a317f65`](https://github.com/bzrudski/vectorose/commit/a317f65218682de2aa800aa1df94b5df23356f42))

- Correct number of frames and set attribute names to match GUI application.
  ([`2829646`](https://github.com/bzrudski/vectorose/commit/2829646e0b64d01b6aeda9e0a28a4f101a6b8db9))

- Correct sphere aspect ratio.
  ([`22b6774`](https://github.com/bzrudski/vectorose/commit/22b6774ce4c4da6a6e86c2e1a117ddcdaa19dabc))

- Fixed type annotation for py 3.9
  ([`a113e09`](https://github.com/bzrudski/vectorose/commit/a113e09e09a7fa7d1b222310ffe6751040fb43ea))

### Build System

- Add dependencies
  ([`cb13c4d`](https://github.com/bzrudski/vectorose/commit/cb13c4d21be567fc67f400a83448b50f6110adce))

- Add development dependencies
  ([`0c18104`](https://github.com/bzrudski/vectorose/commit/0c1810459c6ba164a83ed2e914cf5339696c2c71))

- Add semantic release as dev dependency
  ([`77bf8d8`](https://github.com/bzrudski/vectorose/commit/77bf8d8f6c8bbfbf1937ab2e3b14573988136ce6))

### Documentation

- Begin converting docstrings to numpydoc.
  ([`c79f2f1`](https://github.com/bzrudski/vectorose/commit/c79f2f1dd67896bdce3311feccfa90bad6dca487))

- Correct math formatting and class reference
  ([`6fc0acf`](https://github.com/bzrudski/vectorose/commit/6fc0acf95a39864ca96eeaaf575c321a471025f4))

- Correct warnings and mpl references.
  ([`1ab6f89`](https://github.com/bzrudski/vectorose/commit/1ab6f89d8181f3d9afa328712fd63cc5112b12aa))

- Finish reformatting docs for vectorose.py. Reformat the docs for plotting.py. Edit sphinx
  configuration.
  ([`7b574ec`](https://github.com/bzrudski/vectorose/commit/7b574ec8862191c503b1ebb73cbc6cbe63e880e7))

- Fix whitespace issue in warnings.
  ([`75fc3df`](https://github.com/bzrudski/vectorose/commit/75fc3dfd09c1050a3927a7e7f5413b4ed53e853e))

- Reformat docs in main.py. Add support for todo in docs.
  ([`9c7a28d`](https://github.com/bzrudski/vectorose/commit/9c7a28d3fddc7df1c9eb2d2706527c63e000ae8a))

- Reformat docstrings in vf_io.py.
  ([`0e58c16`](https://github.com/bzrudski/vectorose/commit/0e58c160be5f3cf08afbd8df60c17e8284044a36))

- Reformat documentation for the mock data functions
  ([`a709b4a`](https://github.com/bzrudski/vectorose/commit/a709b4a932948f365aa3597ee1260c564aa97adb))

### Features

- Add ability to export animation.
  ([`e7834d5`](https://github.com/bzrudski/vectorose/commit/e7834d5008357244812906a317285b2dc4312b2d))

- Add function to animate the sphere plot.
  ([`2afadce`](https://github.com/bzrudski/vectorose/commit/2afadce2955110351d16277b0658c320a90ce2c1))

### Refactoring

- Remove unused import.
  ([`2315d80`](https://github.com/bzrudski/vectorose/commit/2315d8062feac63d854909c721d5c7c5f280d825))

- Rename the core module to vectorose.
  ([`f6b991d`](https://github.com/bzrudski/vectorose/commit/f6b991d70aece8f119cc86528e96de4a7bf2eb66))
