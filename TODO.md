# TODO

* Save models
  * After training, save model into `models/{name}/model.pickle` (`model.save(...)`)
  * Test that loading them and predicting works
* Add more features
  * Add features for ...?
* Create final testing python script
  * See `test_model()`
  * Basically train and test all models and write results out
  * No pop-up plots!
* Experiment with creating larger prepared datasets with `max_jitter=50` or so
  * We're not using validation so can make that 0 in the spec
* Define datasets for per-participant splits
* Graphs for the paper
  * Some labeled windows (with onset, window start and end) - both positive and negative examples
