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
```python ga.py smallData.csv CA_Graph_Input_small.csv```

How to run genetic trainer with four-fold cross-fold validation:
```python ga.py smallData.csv CA_Graph_Input_small.csv -f 4```

How to run examples:
```python examples.py smallData.csv CA_Graph_Input_small.csv -e 0```
```python examples.py smallData.csv CA_Graph_Input_small.csv -e 1```
```python examples.py smallData.csv CA_Graph_Input_small.csv -e 2```

# TODO
- [ ] Sripradha: Under Data class in ca.py, create neighborhood graph (Jaime sent you an email)
- [ ] Leland: Visualize ga progress, other fancy pretty things
- [ ] Erika: Update README with usage & expected input information
- [ ] Erika: Comment regression.py & change tabs to spaces
- [ ] Erika: look into ways to evaluate & visualize regression output 'accuracy'
- [ ] Jaime: Keep working on ga.py (crossover, 'selective' perturbation)
