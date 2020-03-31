import pandas as pd
import matplotlib.pyplot as plt
#need to change the filename to the file you want to read
df = pd.read_csv("DQN Train Performance_03-31-00-20.csv")

#choose layout
layout= 'smallGrid'
#layout = 'mediumGrid'


# plot the average game score
plt.clf()
plt.plot(df['Episode End'], df['Average Reward (Score)'], label='Average Game Score Per 100 Episodes')
plt.title('Average Game Score for ' + layout)
plt.savefig('Average Reward for '+ layout + '.png')

# plot the average win rate
plt.clf()
plt.plot(df['Episode End'], df['Average Win Rate'], label='Average Win Rate Per 100 Episodes')
plt.title('Average Win Rate for ' + layout)
plt.savefig('Average Win Rate for '+ layout + '.png')
