# CS760FinalProject
Epidemic prediction using cellular automata
Employing the framework of cellular automata, we set out to predict the oscillating time-series trend in epidemiological data. The key challenges are: is it possible to adopt a non-lattice-based, graph-defined neighborhood structure to learn a cellular automaton? Can we learn an oscillating behavior to predict incident volume n timesteps ahead? Can we chain the predictions to predict volume at one timepoint after another? We find that we can yield high-performing linear models using a graph-based neighborhood that can predict incident volume n timesteps ahead; however, we discover that meeting the last challenge, chaining of predictions to model oscillating behavior, would require a model-evaluation metric capable of both capturing the amplitude and the period in the oscillations and converging onto a very narrow combination of successful parameters in the search space.  

Team:
```
lbhyman@wisc.edu
skarkala@cs.wisc.edu
dlee324@wisc.edu
jawarner2@wisc.edu
```

# Instructions

How to run genetic trainer
```
python ga.py input/smallData.csv input/CA_Graph_Input_small.csv
python ga.py input/smallData.csv input/CA_Graph_Input_small.csv -f 4  
```

How to run examples
```
python examples.py input/smallData.csv input/CA_Graph_Input_small.csv -e 0
python examples.py input/smallData.csv input/CA_Graph_Input_small.csv -e 1
python examples.py input/smallData.csv input/CA_Graph_Input_small.csv -e 2
```
How to run regression trainer
```
python regression.py input/Search_Trend_Flu_Data.csv input/CA_Graph_Input.csv -f 4
```
