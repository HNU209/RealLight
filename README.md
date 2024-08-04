# RealLight

[RealLight](https://github.com/HNU209/RealLight) is a signal optimization algorithm based on reinforcement learning. This algorithm can be used in a wide range of networks at scale and in order to optimize the signal in a realistic way, it obtains the information of the network through CCTV. The proposed RealLight has the advantage that it can be used in a realistic and diverse road network compared to the existing reinforcement learning-based signal optimization algorithm. RealLight's performance verified the best results in terms of vehicle travel time when compared to the existing state-of-the-art algorithm. In addition, RealLight can be used even if each intersection has a different signal table by placing a separate agent. Furthermore, in order to optimize the realistic signal, it is learned not to be involved in the number of lanes or the direction of movement by lanes. The picture below shows the part where the RealLight algorithm can obtain the information of the network through CCTV.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f06d5228-2fed-45e1-9036-66b8b40106b2">
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

## Comparison Table - Average Travel Time
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
    <td align="center">Daejeon-Daeduck</td>
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