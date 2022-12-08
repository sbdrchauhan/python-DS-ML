## Install matplotlib using `pip`:
```bash
pip install matplotlib              # to install for the first time
pip install --upgrade matplotlib    # to upgrade already installed ones
```

## Import package:
```python
import matplotlib.pyplot as plt
import numpy as np    # almost always in need
```

## Style of the Plot:
```python
plt.style.available                # to see what are available styles to choose from
plt.style.use('fivethirtyeight')   # to use one of available styles
```

## 01. Line Plot (Basic):
```python
# developer ages (x values)
dev_age = [25,26,27,28,29,30,31,32,33,34,35]   # list of ages
# median salaries (y values)
dev_salaries = [38496, 42000, 46752, 49320, 53200, 
            56000, 62316, 64928, 67317, 68748, 73752]

# plot it. 'label' argument helps to make legend later
# format strings
# fmt = '[marker][line][color]'. '--k' dash line color black
plt.plot(dev_age, dev_salaries, '--k' label='All Devs')
# same as above but format more explicit
plt.plot(dev_age, dev_salaries, color='k', linestyle='--', marker='*', linewidth=3, label='All Devs')

# now look for python dev
# Median Python Developer Salaries by Age
# same x-range in both plots, could use one
py_dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

# plotting two lines in one plot
# color can be hex color codes: e.g. #5a7d9a, can find online for these
plt.plot(py_dev_x, py_dev_y, color='r', linestyle='-.', marker='o', label='Python Devs')

# add labels to axis and also title
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')

# add legend into the plot
plt.legend()

# to show gridlines
plt.grid(True)

# padding need sometime
plt.tight_layout()

# to save fig
plt.savefig('plot.png')

# to show the plot
plt.show()
```
The final output plot from above code comes out something like this.

<img src="./plot1.png" 
        width="600" 
        height="400" 
        style="display: block; margin: 0 auto" />


## 02. Bar Charts
We can try plotting the same data from above, but now with bar plots placing side-bys-side for different developers. This one is tricky here. We can just replace `plot()` with `bar()` but it will by default put the bars on top of each other. We want them to be side-by-side. We can specify the width of the bars and tell different bar to either shift left or right of that width value, so that when many bars are plotted then they will be shifted and so we can see them all side-by-side.
```python
# need to use numpy array for x values, but similar length as previous
# indexes was needed to make the offsets work
# to get the true x values, later we use xticklabels
x_indexes = np.arange(len(dev_age))
width = 0.25   # default is 0.8 (play to see what you like width to be)

y1_values = dev_salaries      # as above
y2_values = py_dev_salarires  # as above

# now do bar plot
plt.bar(x_indexes - width, y1_values, width=width, color='r', label='All Devs') # notice the shift in x_indexes
plt.bar(x_indexes, y2_values, width=width, color='b', label='Python')

# to make xticks, this will fix to true x values shown
plt.xticks(ticks=x_indexes, labels=dev_age)

# same with most of other stuffs (as above)
```
When you value large lists of values to plot in bar, then vertical bars won't be good, also because the labels in x axis won't have enough room to display as well So, in these scenarios if you need to make the bar plots, then make it **horizontal bar plots**. You just need `barh()` instead of `bar`; and adjust the labels as well. *Read docs for more*.

The final output plot from above code comes out something like this.

<img src="./plot2.png" 
        width="600" 
        height="400" 
        style="display: block; margin: 0 auto" />
