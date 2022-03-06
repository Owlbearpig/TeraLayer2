import numpy as np
np.random.seed(420)

a, b = 0.19737935744311108, 0.300922921527581
#a *= 100
#b *= 100

cs0, cs1, cs2, cs3, ss0, ss1, ss2, ss3 = np.random.random(8)
for trig_f in [cs0, cs1, cs2, cs3, ss0, ss1, ss2, ss3]:
    trig_f *= 1

m_12_r = (0.28920 * cs2 - 0.28920 * cs1)
m_22_r = (0.96104 * cs0 - 0.08702 * cs3)

m_12_i = (- 0.39475 * ss0 - 0.03572 * ss3 + 0.31264 * ss1 - 0.31264 * ss2)
m_22_i = (1.0389 * ss0 + 0.09402 * ss3 + 0.1187 * ss2 - 0.1187 * ss1)

R = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i)

print(R)

#breakpoint()