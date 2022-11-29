# TODO

* Save models
  * Create `models` subdir (gitignored)
  * Give every model a unique name
  * Label every plot with that name
  * After training, save model into `models/{name}/model.pickle` (`model.save(...)`)
  * Test that loading them and predicting works
  * Write all plots as pngs into that directory
  * No pop-up plots!
* Add more features
  * Add features for ...?
* Create final testing python script
  * See `test_model()`
  * Basically train and test all models and write results out
  * No pop-up plots!
* Experiment with creating larger prepared datasets with `max_jitter=50` or so
  * We're not using validation so can make that 0 in the spec
* Define datasets for per-participant splits
