# INFERATOR

IFPEN cpp inference class

Current capabilities :
- inference of MLP generated with TensorFlow (.h5 format only)

Structure:
- Inferator/src: source code of Inferator
- Inferator/env: environment files to use Inferator (_intel ones require VKM-CONVERGE zone access)
- Inferator/test: makefile and simple cpp code for standalone testing (inputs data hardcoded)
- model/wall_functions: models for test case (from E.Rondeaux PhD thesis) in 2 TensorFlow format (.pb and .h5)
- data/wall_functions: .csv file with input data and expect results of test case
 