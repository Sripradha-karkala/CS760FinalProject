# CS760FinalProject
Epidemic prediction using cellular automata

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

# TODO
- [x] Sripradha: Under Data class in ca.py, create neighborhood graph (Jaime sent you an email)
- [x] Leland: Visualize ga progress, other fancy pretty things
- [x] Erika: Update README with usage & expected input information
- [x] Erika: look into ways to evaluate & visualize regression output 'accuracy'
- [ ] Erika: Comment regression.py & change tabs to spaces
- [ ] Jaime: Keep working on ga.py (crossover, 'selective' perturbation)
