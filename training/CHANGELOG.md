# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.5.1](https://github.com/ecmwf/anemoi-core/compare/training-0.5.0...training-0.5.1) (2025-06-17)


### Features

* **models,training:** Shard everything ([#121](https://github.com/ecmwf/anemoi-core/issues/121)) ([06dde94](https://github.com/ecmwf/anemoi-core/commit/06dde94219119746215b767b846542ee31bbff63))


### Bug Fixes

* Config validator checks for conflict with bounding and graphtran… ([#352](https://github.com/ecmwf/anemoi-core/issues/352)) ([f97313e](https://github.com/ecmwf/anemoi-core/commit/f97313e608f0ce2eebd46bad8e5300edebe6eec1))
* **training, losses:** NaN handling for crps ([#358](https://github.com/ecmwf/anemoi-core/issues/358)) ([daf9ddd](https://github.com/ecmwf/anemoi-core/commit/daf9dddda8ff8755b59f0559666d6539d525ba3b))

## [0.5.0](https://github.com/ecmwf/anemoi-core/compare/training-0.4.0...training-0.5.0) (2025-06-05)


### ⚠ BREAKING CHANGES

* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268))
* generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163))
* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52))

### Features

* Generalise activation function ([#163](https://github.com/ecmwf/anemoi-core/issues/163)) ([98c4d06](https://github.com/ecmwf/anemoi-core/commit/98c4d06dfb5b79f605fab885c67a8a4cd6d35384))
* **graphs:** Add scale_resolutions arg ([#188](https://github.com/ecmwf/anemoi-core/issues/188)) ([0ea0642](https://github.com/ecmwf/anemoi-core/commit/0ea06423e4979084b3afe70c30e43bb5dc5f88d5))
* **models,training:** Remove multimapper ([#268](https://github.com/ecmwf/anemoi-core/issues/268)) ([0e8bb99](https://github.com/ecmwf/anemoi-core/commit/0e8bb998176bea2d653ca40772e4e6e1578551f7))
* Transformer mapper ([#179](https://github.com/ecmwf/anemoi-core/issues/179)) ([2cea7db](https://github.com/ecmwf/anemoi-core/commit/2cea7db51d5c5ef63bb4b9c266deb05fb2acf66f))


### Bug Fixes

* Adapt ensemble configs to work with loss refactor.  ([#300](https://github.com/ecmwf/anemoi-core/issues/300)) ([c210478](https://github.com/ecmwf/anemoi-core/commit/c210478d8e84aaa2e93256aa1b4ceda74e191c7d))
* Adapt predict_step in model interface to pass on arguments for model classes ([#281](https://github.com/ecmwf/anemoi-core/issues/281)) ([a5b2643](https://github.com/ecmwf/anemoi-core/commit/a5b26432bc7b78577cd1febd5091b059cc82805c))
* Change output format in validation step to fix issue in plotting ([#305](https://github.com/ecmwf/anemoi-core/issues/305)) ([cc941f3](https://github.com/ecmwf/anemoi-core/commit/cc941f3d7520e230e733654cb45e5c6473b64152))
* Clean params complement ([#291](https://github.com/ecmwf/anemoi-core/issues/291)) ([4e9ca31](https://github.com/ecmwf/anemoi-core/commit/4e9ca313f838f5eccb305d6bf3b9afd8426f095f))
* Dataset_order ([#334](https://github.com/ecmwf/anemoi-core/issues/334)) ([762227a](https://github.com/ecmwf/anemoi-core/commit/762227a5a25843dd4531eef1a9cbe86516eaffcd))
* **deps:** Bump `anemoi-models >= 0.8` ([#351](https://github.com/ecmwf/anemoi-core/issues/351)) ([1fbd525](https://github.com/ecmwf/anemoi-core/commit/1fbd5252ef1400eb4bbe5bfbbb171dad1d652c63))
* **models,training:** Remove unnecessary torch-geometric maximum version ([#326](https://github.com/ecmwf/anemoi-core/issues/326)) ([fe93ea8](https://github.com/ecmwf/anemoi-core/commit/fe93ea8feb379147a9f9e5c5358ea8144855dc77))
* Move scaler to device in scale ([#317](https://github.com/ecmwf/anemoi-core/issues/317)) ([1592d09](https://github.com/ecmwf/anemoi-core/commit/1592d09f0915dddba7dcfc1c11897e0cae5cb6d0))
* Remove activation entry from MLP noise block ([#340](https://github.com/ecmwf/anemoi-core/issues/340)) ([2d060f5](https://github.com/ecmwf/anemoi-core/commit/2d060f5e3382454b06c6369141942b8d6367fb4b))
* Rework Loss Scalings to provide better modularity ([#52](https://github.com/ecmwf/anemoi-core/issues/52)) ([162b906](https://github.com/ecmwf/anemoi-core/commit/162b9062882c321a4a265b0cf561be3f141ac97a))
* **training, models:** Update interpolator to work with new features ([#322](https://github.com/ecmwf/anemoi-core/issues/322)) ([cfdc99f](https://github.com/ecmwf/anemoi-core/commit/cfdc99f984f0038b16cb96d73d02a25284af717e))
* **training:** Bump anemoi-graphs version to 0.5.2 ([#276](https://github.com/ecmwf/anemoi-core/issues/276)) ([9b8ec13](https://github.com/ecmwf/anemoi-core/commit/9b8ec13b56f0a18ac887c754d5d95a7953b2625d))
* **training:** Explicit Batch invariance ([#318](https://github.com/ecmwf/anemoi-core/issues/318)) ([45f6e15](https://github.com/ecmwf/anemoi-core/commit/45f6e15183dbf61ad259b7ca5ed35d94395d237c))
* **training:** No logging variable metadata to mlflow ([#304](https://github.com/ecmwf/anemoi-core/issues/304)) ([4f7c2c9](https://github.com/ecmwf/anemoi-core/commit/4f7c2c9c75ae6e67a148699a317528cd7651bf46))
* Update defaults for mlflow logging ([#333](https://github.com/ecmwf/anemoi-core/issues/333)) ([5560581](https://github.com/ecmwf/anemoi-core/commit/556058167439ae29e6c9559d67f2e01df8466158))


### Documentation

* Downloading era5 o96 dataset ([#307](https://github.com/ecmwf/anemoi-core/issues/307)) ([b9e2d24](https://github.com/ecmwf/anemoi-core/commit/b9e2d24702132e43f160a428ccad88b7584fcabe))
* Fix minor mistakes in CRPS user guide. ([#264](https://github.com/ecmwf/anemoi-core/issues/264)) ([52d9a2e](https://github.com/ecmwf/anemoi-core/commit/52d9a2e5e0cef82b65b604ed2dc55016a48aeaa2))

## [0.4.0](https://github.com/ecmwf/anemoi-core/compare/training-0.3.3...training-0.4.0) (2025-04-16)


### ⚠ BREAKING CHANGES

* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153))
* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79))

### Features

* Add a CLI to dump the Hydra configuration files into a single YAML file. ([#137](https://github.com/ecmwf/anemoi-core/issues/137)) ([ef1e76e](https://github.com/ecmwf/anemoi-core/commit/ef1e76e2e15bb412adb184e1b33e003590c72e8a))
* Add EarlyStopping Wrapper ([#130](https://github.com/ecmwf/anemoi-core/issues/130)) ([21d06be](https://github.com/ecmwf/anemoi-core/commit/21d06be94b5ea09889777038da749ee167cf3f3d))
* Add the possibility to train a model with a dry MLflow run ID ([#164](https://github.com/ecmwf/anemoi-core/issues/164)) ([9849d21](https://github.com/ecmwf/anemoi-core/commit/9849d211fcb03fcbb76fcdd3ef6e97e86f1e69b4))
* **config:** Improved configuration and data structures ([#79](https://github.com/ecmwf/anemoi-core/issues/79)) ([1f7812b](https://github.com/ecmwf/anemoi-core/commit/1f7812b559b51d842852df29ace7dda6d0f66ef2))
* edge post-processor ([#199](https://github.com/ecmwf/anemoi-core/issues/199)) ([1450de7](https://github.com/ecmwf/anemoi-core/commit/1450de739be9988cdb23fbdb23a0463859066e7c))
* **graphs:** migrate edge builders to torch-cluster ([#56](https://github.com/ecmwf/anemoi-core/issues/56)) ([f67da66](https://github.com/ecmwf/anemoi-core/commit/f67da664c18762e4c8a8cf0af9d4e97ec7315454))
* Kcrps  ([#182](https://github.com/ecmwf/anemoi-core/issues/182)) ([8bbe898](https://github.com/ecmwf/anemoi-core/commit/8bbe89839e2eff3fcbc35613eb92920d4afc3276))
* make colormaps configurable for groups of variable ([#124](https://github.com/ecmwf/anemoi-core/issues/124)) ([83d72e1](https://github.com/ecmwf/anemoi-core/commit/83d72e17b5c8017a1d9a47f75e0848e0bbb080a3))
* **mlflow:** Make direct endpoint calls robust ([#186](https://github.com/ecmwf/anemoi-core/issues/186)) ([77bd890](https://github.com/ecmwf/anemoi-core/commit/77bd890004ea591861ec1efe6732ea1599008985))
* **models,training:** temporal interpolation ([#153](https://github.com/ecmwf/anemoi-core/issues/153)) ([ea644ce](https://github.com/ecmwf/anemoi-core/commit/ea644ce1c9aef902333d9cbb30bcde0a3746fbcc))
* **models:** adding leaky boundings ([#256](https://github.com/ecmwf/anemoi-core/issues/256)) ([426e860](https://github.com/ecmwf/anemoi-core/commit/426e86048d6c0a03750fb0e205890841c27c8148))
* **training:** Add initial TimeLimit callback ([#115](https://github.com/ecmwf/anemoi-core/issues/115)) ([41ff583](https://github.com/ecmwf/anemoi-core/commit/41ff5830dc0ba08aaa86e052a1a0aac8c4498c7a))


### Bug Fixes

* --longtests not available ([#200](https://github.com/ecmwf/anemoi-core/issues/200)) ([9dfec0a](https://github.com/ecmwf/anemoi-core/commit/9dfec0a3bd2043e646cc49b5302fcc4d669e4a41))
* Checkpoint path check for multiple tasks/GPUs training ([#242](https://github.com/ecmwf/anemoi-core/issues/242)) ([449f8bd](https://github.com/ecmwf/anemoi-core/commit/449f8bd6044043fc46160b602762785b475cd1ce))
* **configs,schemas:** hierarchical schemas ([#221](https://github.com/ecmwf/anemoi-core/issues/221)) ([2d4a54d](https://github.com/ecmwf/anemoi-core/commit/2d4a54d3c6ed2d41fd6cbf2ef3ac57b9efb2f968))
* correct config comment regarding config_validation flag ([#245](https://github.com/ecmwf/anemoi-core/issues/245)) ([d02e0bb](https://github.com/ecmwf/anemoi-core/commit/d02e0bb2a9bf1700de90c8d862f80a510f03eceb))
* dataset schema too defined too strictly ([#143](https://github.com/ecmwf/anemoi-core/issues/143)) ([4792ee1](https://github.com/ecmwf/anemoi-core/commit/4792ee145a3bda05f934180485ddb7b6e3bb25d4))
* datashader new release test ([#104](https://github.com/ecmwf/anemoi-core/issues/104)) ([e9c0701](https://github.com/ecmwf/anemoi-core/commit/e9c0701bce9e5dfff7184fc6210092b82b9386a6))
* dry run forking ([#260](https://github.com/ecmwf/anemoi-core/issues/260)) ([a32cccd](https://github.com/ecmwf/anemoi-core/commit/a32cccd3179b049979b500ad878480ae5e46d050))
* **graphs,training:** skip hidden attributes ([#176](https://github.com/ecmwf/anemoi-core/issues/176)) ([468c45a](https://github.com/ecmwf/anemoi-core/commit/468c45a8a8ccee6f4a11c4ddf3fe4a7220235d08))
* lam rollout ([#213](https://github.com/ecmwf/anemoi-core/issues/213)) ([6f78387](https://github.com/ecmwf/anemoi-core/commit/6f78387da08237b7a63b84575e8cb6c19b254493))
* pydantic schemas move ([#228](https://github.com/ecmwf/anemoi-core/issues/228)) ([6bca9bc](https://github.com/ecmwf/anemoi-core/commit/6bca9bc66ff54ac294d97793b8cebed1cd1bb8a4))
* remove pydantic schemas from checkpoint pickled objects ([#237](https://github.com/ecmwf/anemoi-core/issues/237)) ([ecb945a](https://github.com/ecmwf/anemoi-core/commit/ecb945ad7df226d16ed42b10123372741c5748a5))
* Rename model_run_ids as trajectory_ids ([#216](https://github.com/ecmwf/anemoi-core/issues/216)) ([e5e942d](https://github.com/ecmwf/anemoi-core/commit/e5e942d001640b73a29d5d6156364f3bd99fc1e5))
* **training,configs:** update enc-dec config ([#125](https://github.com/ecmwf/anemoi-core/issues/125)) ([beb8c69](https://github.com/ecmwf/anemoi-core/commit/beb8c69e7c58ae04124030ddf3b080b9c4e6b7e1))
* **training,schema:** error when using ReweightedGraphNodeAttribute ([#169](https://github.com/ecmwf/anemoi-core/issues/169)) ([a6313ef](https://github.com/ecmwf/anemoi-core/commit/a6313ef37ec8e08fe0fbcc58b99fb098774b0229))
* **training:** added weights_only=False flag to torch.load to avoid crashes in torch 2.6 ([#205](https://github.com/ecmwf/anemoi-core/issues/205)) ([02b5117](https://github.com/ecmwf/anemoi-core/commit/02b5117fb2f455fea943ffccc76117d2e0d514f8))
* **training:** Rework Combined Loss ([#103](https://github.com/ecmwf/anemoi-core/issues/103)) ([b63f1aa](https://github.com/ecmwf/anemoi-core/commit/b63f1aa4e6f154898d84310cb03cf244b322efa4))
* update stretched.yaml config to be consistent with schema ([#195](https://github.com/ecmwf/anemoi-core/issues/195)) ([21255ac](https://github.com/ecmwf/anemoi-core/commit/21255ac8e535889ad47ccada2bf4a27047c27021))
* update to null and bump versions ([#263](https://github.com/ecmwf/anemoi-core/issues/263)) ([b4507fb](https://github.com/ecmwf/anemoi-core/commit/b4507fbf926a01f0e042cb61fc35b3901243ddc0))
* updates to schemas to correct minor pydantic issues ([#144](https://github.com/ecmwf/anemoi-core/issues/144)) ([4e51c17](https://github.com/ecmwf/anemoi-core/commit/4e51c178c72c6853cf223e81609e0a91b183a277))


### Documentation

* Add diagnostics to schema docs ([#159](https://github.com/ecmwf/anemoi-core/issues/159)) ([42c5706](https://github.com/ecmwf/anemoi-core/commit/42c570625cfe445441f18e74827b8c1526ff1782))
* Add reference in training getting started guide for intersphinx ([#220](https://github.com/ecmwf/anemoi-core/issues/220)) ([f1d5e1f](https://github.com/ecmwf/anemoi-core/commit/f1d5e1f07c7a92e7152f0a392a046537bc718cab))
* Add subheadings to schema doc page ([#149](https://github.com/ecmwf/anemoi-core/issues/149)) ([d3c7de9](https://github.com/ecmwf/anemoi-core/commit/d3c7de905bced2dc9e75a92de4e9abf848936e62))
* fix documentation to refer to anemoi datasets instead of zarr datasets ([#154](https://github.com/ecmwf/anemoi-core/issues/154)) ([ad062b2](https://github.com/ecmwf/anemoi-core/commit/ad062b22cdd05354bc010eabbf8ffa806def081c))
* **models:** Docathon  ([#202](https://github.com/ecmwf/anemoi-core/issues/202)) ([5dba9d3](https://github.com/ecmwf/anemoi-core/commit/5dba9d34d65d4331dabd19355c7a31f7f1468fbf))
* **training:** add ADR into docs/ ([#168](https://github.com/ecmwf/anemoi-core/issues/168)) ([ee4da88](https://github.com/ecmwf/anemoi-core/commit/ee4da886c32f1edbc197215780bf84dcd323e172))
* **training:** Docathon ([#201](https://github.com/ecmwf/anemoi-core/issues/201)) ([e69430f](https://github.com/ecmwf/anemoi-core/commit/e69430f8c1ba8e7de50cd99f202e3f4876b806e0))
* Update docs for kcrps. ([#258](https://github.com/ecmwf/anemoi-core/issues/258)) ([79cbd1d](https://github.com/ecmwf/anemoi-core/commit/79cbd1d5e5f0f5aa82ce712bed474a6ad99f17e8))
* use new logo ([#140](https://github.com/ecmwf/anemoi-core/issues/140)) ([c269cea](https://github.com/ecmwf/anemoi-core/commit/c269cea3c84f2e35ef0a318e0cd1b769d285177c))

## [0.3.3](https://github.com/ecmwf/anemoi-core/compare/training-0.3.2...training-0.3.3) (2025-02-05)


### Features

* make flash attention configurable ([#60](https://github.com/ecmwf/anemoi-core/issues/60)) ([41fcab6](https://github.com/ecmwf/anemoi-core/commit/41fcab6335b334fdbebeb944c904cdbea6388889))
* Model Freezing ❄️  ([#61](https://github.com/ecmwf/anemoi-core/issues/61)) ([54e42cf](https://github.com/ecmwf/anemoi-core/commit/54e42cf42a47e00be96b464f501f6a462cd1bca4))
* **models:** normalization layers ([#47](https://github.com/ecmwf/anemoi-core/issues/47)) ([0e1c7c4](https://github.com/ecmwf/anemoi-core/commit/0e1c7c4840138debf877bb954b45f4c3a1cd0e33))


### Bug Fixes

* cancel RTD builds on no change ([#97](https://github.com/ecmwf/anemoi-core/issues/97)) ([36522d8](https://github.com/ecmwf/anemoi-core/commit/36522d87cdd95a5cb54b4c865eca67a64e22fffa))
* only load shards of grid into cpu mem if possible ([#83](https://github.com/ecmwf/anemoi-core/issues/83)) ([abbef4b](https://github.com/ecmwf/anemoi-core/commit/abbef4bf505af553584d9e0d4825db544d1c9ca7))
* pin dask version to 2024.12.1  ([#94](https://github.com/ecmwf/anemoi-core/issues/94)) ([074c0f2](https://github.com/ecmwf/anemoi-core/commit/074c0f226bf6541df977e43eff626a68d2f1f1fe))
* **training:** profiler 'Model Summary' works when sharding models over multiple GPUs ([#90](https://github.com/ecmwf/anemoi-core/issues/90)) ([9d9e89a](https://github.com/ecmwf/anemoi-core/commit/9d9e89a8fd6b90ecd62f799d942c84a4e984b9ed))
* update configs to avoid DeprecationWarning ([#53](https://github.com/ecmwf/anemoi-core/issues/53)) ([3560290](https://github.com/ecmwf/anemoi-core/commit/356029039e8406bc02a2b22f56107babbf2e7551))


### Documentation

* **graphs:** Refactor anemoi-graphs documentation ([#49](https://github.com/ecmwf/anemoi-core/issues/49)) ([29942b6](https://github.com/ecmwf/anemoi-core/commit/29942b6c088c7a2c6ff3f8ac13277041086cda9f))
* Improve installation docs ([#91](https://github.com/ecmwf/anemoi-core/issues/91)) ([0b5f8fb](https://github.com/ecmwf/anemoi-core/commit/0b5f8fb8b93555d76ebe3316c430121350bf5243))
* point RTD to right subfolder ([5a80cb6](https://github.com/ecmwf/anemoi-core/commit/5a80cb6047e864ea97bed06a76ddc54507e5fcbe))
* Tidy for core ([b24c521](https://github.com/ecmwf/anemoi-core/commit/b24c521c447272afd1b209745b24d16794cdb85a))

## [Unreleased](https://github.com/ecmwf/anemoi-training/compare/0.3.2...HEAD)

### Fixed

- Profilers 'Model Summary' feature works when the model is sharded across GPUs [#90](https://github.com/ecmwf/anemoi-core/pull/90)

## [0.3.2 - Multiple Fixes, Checkpoint updates, Stretched-grid/LAM updates](https://github.com/ecmwf/anemoi-training/compare/0.3.1...0.3.2) - 2024-12-19

### Fixed

- Not update NaN-weight-mask for loss function when using remapper and no imputer [#178](https://github.com/ecmwf/anemoi-training/pull/178)
- Dont crash when using the profiler if certain env vars arent set [#180](https://github.com/ecmwf/anemoi-training/pull/180)
- Remove saving of metadata to training checkpoint [#57](https://github.com/ecmwf/anemoi-training/pull/190)
- Fixes to callback plots [#182] (power spectrum large numpy array error + precip cmap for cases where precip is prognostic).
- GraphTrainableParameters callback will log a warning when no trainable parameters are specified  [#173](https://github.com/ecmwf/anemoi-training/pull/173)
- Fixes to checkpoint saving - ensure last checkpoint if saving when using max_steps [#191] (https://github.com/ecmwf/anemoi-training/pull/191)
- Identify stretched grid models based on graph rather than configuration file [#204](https://github.com/ecmwf/anemoi-training/pull/204)

### Added

- Introduce variable to configure: transfer_learning -> bool, True if loading checkpoint in a transfer learning setting.
-
<b> TRANSFER LEARNING</b>: enabled new functionality. You can now load checkpoints from different models and different training runs.
- Introduce (optional) variable to configure: transfer_learning -> bool, True if loading checkpoint in a transfer learning setting.
- <b> TRANSFER LEARNING</b>: enabled new functionality. You can now load checkpoints from different models and different training runs.
- Effective batch size: `(config.dataloader.batch_size["training"] * config.hardware.num_gpus_per_node * config.hardware.num_nodes) // config.hardware.num_gpus_per_model`.
  Used for experiment reproducibility across different computing configurations.
- Added a check for the variable sorting on pre-trained/finetuned models [#120](https://github.com/ecmwf/anemoi-training/pull/120)
- Added default configuration files for stretched grid and limited area model experiments [173](https://github.com/ecmwf/anemoi-training/pull/173)
- Added new metrics for stretched grid models to track losses inside/outside the regional domain [#199](https://github.com/ecmwf/anemoi-training/pull/199)
- <b> Model Freezing ❄️</b>: enabled new functionality. You can now Freeze parts of your model by specifying a list of submodules to freeze with the new config parameter: submodules_to_freeze.
- Introduce (optional) variable to configure: submodules_to_freeze -> List[str], list of submodules to freeze.
- Add supporting arrrays (numpy) to checkpoint
- Support for masking out unconnected nodes in LAM [#171](https://github.com/ecmwf/anemoi-training/pull/171)
- Improved validation metrics, allow 'all' to be scaled [#202](https://github.com/ecmwf/anemoi-training/pull/202)
- Added default configuration files for hierarchical processor [175](https://github.com/ecmwf/anemoi-training/pull/175)

### Changed

### Removed

- Removed the resolution config entry [#120](https://github.com/ecmwf/anemoi-training/pull/120)

## [0.3.1 - AIFS v0.3 Compatibility](https://github.com/ecmwf/anemoi-training/compare/0.3.0...0.3.1) - 2024-11-28

### Changed

- Perform full shuffle of training dataset [#153](https://github.com/ecmwf/anemoi-training/pull/153)

### Fixed

- Update `n_pixel` used by datashader to better adapt across resolutions [#152](https://github.com/ecmwf/anemoi-training/pull/152)
- Fixed bug in power spectra plotting for the n320 resolution.
- Allow histogram and spectrum plot for one variable [#165](https://github.com/ecmwf/anemoi-training/pull/165)

### Added

- Introduce variable to configure (Cosine Annealing) optimizer warm up [#155](https://github.com/ecmwf/anemoi-training/pull/155)
- Add reader groups to reduce CPU memory usage and increase dataloader throughput [#76](https://github.com/ecmwf/anemoi-training/pull/76)
- Bump `anemoi-graphs` version to 0.4.1 [#159](https://github.com/ecmwf/anemoi-training/pull/159)

## [0.3.0 - Loss & Callback Refactors](https://github.com/ecmwf/anemoi-training/compare/0.2.2...0.3.0) - 2024-11-14

### Fixed

- Rename loss_scaling to variable_loss_scaling [#138](https://github.com/ecmwf/anemoi-training/pull/138)
- Refactored callbacks. [#60](https://github.com/ecmwf/anemoi-training/pulls/60)
  - Updated docs [#115](https://github.com/ecmwf/anemoi-training/pull/115)
  - Fix enabling LearningRateMonitor [#119](https://github.com/ecmwf/anemoi-training/pull/119)

- Refactored rollout [#87](https://github.com/ecmwf/anemoi-training/pulls/87)
  - Enable longer validation rollout than training

- Expand iterables in logging [#91](https://github.com/ecmwf/anemoi-training/pull/91)
  - Save entire config in mlflow


### Added

- Included more loss functions and allowed configuration [#70](https://github.com/ecmwf/anemoi-training/pull/70)
- Include option to use datashader and optimised asyncronohous callbacks [#102](https://github.com/ecmwf/anemoi-training/pull/102)
  - Fix that applies the metric_ranges in the post-processed variable space [#116](https://github.com/ecmwf/anemoi-training/pull/116)

- Allow updates to scalars [#137](https://github.com/ecmwf/anemoi-training/pulls/137)
  - Add without subsetting in ScaleTensor

- Sub-hour datasets [#63](https://github.com/ecmwf/anemoi-training/pull/63)
- Add synchronisation workflow [#92](https://github.com/ecmwf/anemoi-training/pull/92)
- Feat: Anemoi Profiler compatible with mlflow and using Pytorch (Kineto) Profiler for memory report [38](https://github.com/ecmwf/anemoi-training/pull/38/)
- Feat: Save a gif for longer rollouts in validation [#65](https://github.com/ecmwf/anemoi-training/pull/65)
- New limited area config file added, limited_area.yaml. [#134](https://github.com/ecmwf/anemoi-training/pull/134/)
- New stretched grid config added, stretched_grid.yaml [#133](https://github.com/ecmwf/anemoi-training/pull/133)
- Functionality to change the weight attribute of nodes in the graph at the start of training without re-generating the graph. [#136] (https://github.com/ecmwf/anemoi-training/pull/136)
- Custom System monitor for Nvidia and AMD GPUs [#147](https://github.com/ecmwf/anemoi-training/pull/147)

### Changed

- Renamed frequency keys in callbacks configuration. [#118](https://github.com/ecmwf/anemoi-training/pull/118)
- Modified training configuration to support max_steps and tied lr iterations to max_steps by default [#67](https://github.com/ecmwf/anemoi-training/pull/67)
- Merged node & edge trainable feature callbacks into one. [#135](https://github.com/ecmwf/anemoi-training/pull/135)
- Increase the default MlFlow HTTP max retries [#111](https://github.com/ecmwf/anemoi-training/pull/111)

### Removed

## [0.2.2 - Maintenance: pin python <3.13](https://github.com/ecmwf/anemoi-training/compare/0.2.1...0.2.2) - 2024-10-28

### Changed

- Lock python version <3.13 [#107](https://github.com/ecmwf/anemoi-training/pull/107)

## [0.2.1 - Bugfix: resuming mlflow runs](https://github.com/ecmwf/anemoi-training/compare/0.2.0...0.2.1) - 2024-10-24

### Added

- Mlflow-sync to include new tag for server to server syncing [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Mlflow-sync to include functionality to resume and fork server2server runs [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Rollout training for Limited Area Models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)
- Feature: New `Boolean1DMask` class. Enables rollout training for limited area models. [#79](https://github.com/ecmwf/anemoi-training/pulls/79)

### Fixed

- Fix pre-commit regex
- Mlflow-sync to handle creation of new experiments in the remote server [#83] (https://github.com/ecmwf/anemoi-training/pull/83)
- Fix for multi-gpu when using mlflow due to refactoring of _get_mlflow_run_params function [#99] (https://github.com/ecmwf/anemoi-training/pull/99)
- ci: fix pyshtools install error (#100) https://github.com/ecmwf/anemoi-training/pull/100
- Mlflow-sync to handle creation of new experiments in the remote server [#83](https://github.com/ecmwf/anemoi-training/pull/83)
- Fix for multi-gpu when using mlflow due to refactoring of _get_mlflow_run_params function [#99](https://github.com/ecmwf/anemoi-training/pull/99)
- ci: fix pyshtools install error [#100](https://github.com/ecmwf/anemoi-training/pull/100)
- Fix `__version__` import in init

### Changed

- Update copyright notice

## [0.2.0 - Feature release](https://github.com/ecmwf/anemoi-training/compare/0.1.0...0.2.0) - 2024-10-16

- Make pin_memory of the Dataloader configurable (#64)

### Added

- Add anemoi-transform link to documentation
- Codeowners file (#56)
- Changelog merge strategy (#56)
- Contributors file (#106)

#### Miscellaneous

- Introduction of remapper to anemoi-models leads to changes in the data indices. Some preprocessors cannot be applied in-place anymore.

- Variable Bounding as configurable model layers [#13](https://github.com/ecmwf/anemoi-models/issues/13)


#### Functionality

- Enable the callback for plotting a histogram for variables containing NaNs
- Enforce same binning for histograms comparing true data to predicted data
- Fix: Inference checkpoints are now saved according the frequency settings defined in the config [#37](https://github.com/ecmwf/anemoi-training/pull/37)
- Feature: Add configurable models [#50](https://github.com/ecmwf/anemoi-training/pulls/50)
- Feature: Authentication support for mlflow sync - [#51](https://github.com/ecmwf/anemoi-training/pull/51)
- Feature: Support training for datasets with missing time steps [#48](https://github.com/ecmwf/anemoi-training/pulls/48)
- Feature: `AnemoiMlflowClient`, an mlflow client with authentication support [#86](https://github.com/ecmwf/anemoi-training/pull/86)
- Long Rollout Plots
- Mask NaN values in training loss function [#72](https://github.com/ecmwf/anemoi-training/pull/72) and [#271](https://github.com/ecmwf-lab/aifs-mono/issues/271)

### Fixed

- Fix `TypeError` raised when trying to JSON serialise `datetime.timedelta` object - [#43](https://github.com/ecmwf/anemoi-training/pull/43)
- Bugfixes for CI (#56)
- Fix `mlflow` subcommand on python 3.9 [#62](https://github.com/ecmwf/anemoi-training/pull/62)
- Show correct subcommand in MLFlow - Addresses [#39](https://github.com/ecmwf/anemoi-training/issues/39) in [#61](https://github.com/ecmwf/anemoi-training/pull/61)
- Fix interactive multi-GPU training [#82](https://github.com/ecmwf/anemoi-training/pull/82)
- Allow 500 characters in mlflow logging [#88](https://github.com/ecmwf/anemoi-training/pull/88)

### Changed

- Updated configuration examples in documentation and corrected links - [#46](https://github.com/ecmwf/anemoi-training/pull/46)
- Remove credential prompt from mlflow login, replace with seed refresh token via web - [#78](https://github.com/ecmwf/anemoi-training/pull/78)
- Update CODEOWNERS
- Change how mlflow measures CPU Memory usage - [94](https://github.com/ecmwf/anemoi-training/pull/94)

## [0.1.0 - Anemoi training - First release](https://github.com/ecmwf/anemoi-training/releases/tag/0.1.0) - 2024-08-16

### Added

#### Subcommands

- Subcommand for training `anemoi-training train`
- Subcommand for config generation of configs
- Subcommand for mlflow: login and sync
- Subcommand for checkpoint handling

#### Functionality

- Searchpaths for Hydra configs, to enable configs in CWD, `ANEMOI_CONFIG_PATH` env, and `.config/anemoi/training` in addition to package defaults
- MlFlow token authentication
- Configurable pressure level scaling

#### Continuous Integration / Deployment

- Downstream CI to test all dependencies with changes
- Changelog Status check
- Readthedocs PR builder
- Changelog Release Updater Workflow

#### Miscellaneous

- Extended ruff Ruleset
- Added Docsig pre-commit hook
- `__future__` annotations for typehints
- Added Typehints where missing
- Added Changelog
- Correct errors in callback plots
- fix error in the default config
- example slurm config
- ability to configure precip-type plots

### Changed

#### Move to Anemoi Ecosystem

- Fixed PyPI packaging
- Use of Anemoi models
- Use of Anemoi graphs
- Adjusted tests to work with new Anemoi ecosystem
- Adjusted configs to reasonable common defaults

#### Functionality

- Changed hardware-specific keys from configs to `???` to trigger "missing"
- `__len__` of NativeGridDataset
- Configurable dropout in attention layer

#### Docs

- First draft on Read the Docs
- Fixed docstrings

#### Miscellaneous

- Moved callbacks into folder to fascilitate future refactor
- Adjusted PyPI release infrastructure to common ECMWF workflow
- Bumped versions in Pre-commit hooks
- Fix crash when logging hyperparameters with missing values in the config
- Fixed "null" tracker metadata when tracking is disabled, now returns an empty dict
- Pinned numpy<2 until we can test all migration
- (ci): path ignore of docs for downstream ci
- (ci): remove yaml anchor, unsupported by Github
- ci: make python QA reusable
- ci: permissions on changelog updater

### Removed

- Dependency on mlflow-export-import
- Specific user configs
- **len** function of NativeGridDataset as it lead to bugs

<!-- Add Git Diffs for Links above -->
