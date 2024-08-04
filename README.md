# RealLight

[RealLight](https://github.com/HNU209/RealLight) is a signal optimization algorithm based on reinforcement learning. This algorithm is large and can be used in a wide range of networks. The proposed RealLight has the advantage of being realistic compared to existing reinforcement learning-based signal optimization algorithms and can be used in a variety of road networks. RealLight's performance verified the best results in terms of vehicle travel time compared to existing state-of-the-art algorithms. RealLight also has individual agents at each intersection so that it can be utilized even if each intersection has a different signal table. Furthermore, it is learned not to be involved in the number of lanes or the direction of movement by lane in order to optimize realistic signals.

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
<table style="text-align:center">
  <tr>
    <td colspan="2">Env</td>
    <td>1x3</td>
    <td>2x2</td>
    <td>3x3</td>
    <td>4x4</td>
    <td>Jinan</td>
    <td>Hangzhou</td>
    <td>Newyork</td>
    <td>Daejeon-Daeduck</td>
  </tr>
  <tr>
    <td rowspan="2">Non-RL</td>
    <td>Fixed-Time</td>
    <td>384.47</td>
    <td>454.01</td>
    <td>508.87</td>
    <td>565.99</td>
    <td>405.91</td>
    <td>488.51</td>
    <td>-</td>
    <td>207.45</td>
  </tr>
    <td>SOTL</td>
    <td>247.07</td>
    <td>331.64</td>
    <td>424.66</td>
    <td>474.32</td>
    <td>410.65</td>
    <td>505.53</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">RL</td>
    <td>GRL</td>
    <td>208.21</td>
    <td>239.13</td>
    <td>431.43</td>
    <td>523.01</td>
    <td>562.91</td>
    <td>598.17</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Colight</td>
    <td>210.01</td>
    <td>312.29</td>
    <td>328.7</td>
    <td>397.07</td>
    <td>327.62</td>
    <td>337.45</td>
    <td>1459.28</td>
    <td>-</td>
  </tr>
  <tr>
    <td>PressLight</td>
    <td>98.74</td>
    <td>123.9</td>
    <td>166.28</td>
    <td>215.32</td>
    <td>285.65</td>
    <td>341.99</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>IPDALight</td>
    <td>88.01</td>
    <td>109.66</td>
    <td>146.92</td>
    <td>184.54</td>
    <td>255.35</td>
    <td>298.99</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="background-color:PeachPuff">
    <td rowspan="2">Ours</td>
    <td>RealLight</td>
    <td>85.71</td>
    <td>107.08</td>
    <td>142.83</td>
    <td>181.39</td>
    <td>253.39</td>
    <td>298.19</td>
    <td>887.82</td>
    <td>122.27</td>
  </tr>
    <td>RealLight-max</td>
    <td>89.43</td>
    <td>113.45</td>
    <td>151.63</td>
    <td>193.63</td>
    <td>271.34</td>
    <td>319.57</td>
    <td>931.52</td>
    <td>129.68</td>
</table>