# RealLight

[RealLight](https://github.com/HNU209/RealLight) is a reinforcement learning-based signal optimization algorithm designed to operate effectively across large-scale and complex road networks. This algorithm can be used in a wide range of networks at scale and in order to optimize the signal in a realistic way, it obtains the information of the network through surveillance cameras. Compared to existing reinforcement learning-based traffic signal control algorithms, RealLight offers greater applicability to diverse and realistic urban networks. Its performance has been validated by demonstrating superior results in terms of vehicle travel time against state-of-the-art algorithm. A Key advantage of RealLight is its flexibility: it supports heterogeneous signal settings by deploying a separate agent at each intersection, allowing it to handle intersections with different signal tables. Moreover, RealLight is trained to optimize traffic signals without relying on specific lane counts or directional movements, enhancing its generalization.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f06d5228-2fed-45e1-9036-66b8b40106b2" width=500px height=300px>
  <img src="figure/seo_gu.gif" width=300px height=300px>
</p>
<p align="center">
</p>

## How to use RealLight

### Requirements
- Python 3.8+
- [CityFlow](https://github.com/cityflow-project/CityFlow)
- pytorch 2.1.0+
- numpy
- yaml
- pandas

### Getting Started
1. Clone RealLight
    ```
    git clone https://github.com/HNU209/DTUMOS.git
    ```

2. Run run.py
    ```
    python run.py
    ```

### Adjust custom hyper-parameters
If you want to adjust your hyper-parameters, modify conf.yaml file

## A comparison table by Average Travel Time
<table>
  <tr>
    <td colspan="2" align="center">Environments</td>
    <td align="center">1x3</td>
    <td align="center">2x2</td>
    <td align="center">3x3</td>
    <td align="center">4x4</td>
    <td align="center">Jinan</td>
    <td align="center">Hangzhou</td>
    <td align="center">Newyork</td>
    <td align="center">Daejeon Seo-gu</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">Non-RL</td>
    <td align="center">Fixed-Time</td>
    <td align="center">384.47</td>
    <td align="center">454.01</td>
    <td align="center">508.87</td>
    <td align="center">565.99</td>
    <td align="center">405.91</td>
    <td align="center">488.51</td>
    <td align="center">-</td>
    <td align="center">207.45</td>
  </tr>
    <td align="center">SOTL</td>
    <td align="center">247.07</td>
    <td align="center">331.64</td>
    <td align="center">424.66</td>
    <td align="center">474.32</td>
    <td align="center">410.65</td>
    <td align="center">505.53</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="4" align="center">RL</td>
    <td align="center">GRL</td>
    <td align="center">208.21</td>
    <td align="center">239.13</td>
    <td align="center">431.43</td>
    <td align="center">523.01</td>
    <td align="center">562.91</td>
    <td align="center">598.17</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">Colight</td>
    <td align="center">210.01</td>
    <td align="center">312.29</td>
    <td align="center">328.7</td>
    <td align="center">397.07</td>
    <td align="center">327.62</td>
    <td align="center">337.45</td>
    <td align="center">1459.28</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">PressLight</td>
    <td align="center">98.74</td>
    <td align="center">123.9</td>
    <td align="center">166.28</td>
    <td align="center">215.32</td>
    <td align="center">285.65</td>
    <td align="center">341.99</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">IPDALight</td>
    <td align="center">88.01</td>
    <td align="center">109.66</td>
    <td align="center">146.92</td>
    <td align="center">184.54</td>
    <td align="center">255.35</td>
    <td align="center">298.99</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">Ours</td>
    <td align="center">RealLight</td>
    <td align="center">85.71</td>
    <td align="center">107.08</td>
    <td align="center">142.83</td>
    <td align="center">181.39</td>
    <td align="center">253.39</td>
    <td align="center">298.19</td>
    <td align="center">887.82</td>
    <td align="center">122.27</td>
  </tr>
    <td align="center">RealLight-max</td>
    <td align="center">89.43</td>
    <td align="center">113.45</td>
    <td align="center">151.63</td>
    <td align="center">193.63</td>
    <td align="center">271.34</td>
    <td align="center">319.57</td>
    <td align="center">931.52</td>
    <td align="center">129.68</td>
</table>