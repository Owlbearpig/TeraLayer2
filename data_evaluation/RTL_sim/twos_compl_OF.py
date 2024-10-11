from twos_compl_datatype import Bin2sComp, one_as, zero_as, one, zero, trunc
from numpy import pi as pi64
from scratches.snippets.base_converters import dec_to_twoscompl
from numba import jit

def cost(point, pd, p):
    """
    // p = [239.777814149857 476.259423971176 235.382882833481]
    // => f(p_sol, p) = 8.00341041043292 / 2 = 4.00170520521646 (python)

    // model data (r_exp) for p_sol = [168. 609.  98.],
    cur_data_real = {
        25'b000_0100010010100110110101, // (0.2681706279533744+0.1967872201553877j)
        25'b000_0110001001011101000101, // (0.38423295053796597+0.09832108527416124j)
        25'b000_1001000100100100000011, // (0.5669563654732931-0.0262901022740048j)
        25'b000_0100111010011011101010, // (0.3070627100951053+0.02696250364705963j)
        25'b111_1000011011100101010111, // (-0.4730629191512655-0.07977137762215616j)
        25'b000_0100010010001111000111 // (0.267808722530607+0.4760014734738304j)
    };
    cur_data_imag = {
        25'b000_0011001001100000101001, // (0.2681706279533744+0.1967872201553877j)
        25'b000_0001100100101011100100, // (0.38423295053796597+0.09832108527416124j)
        25'b111_1111100101000101000100, // (0.5669563654732931-0.0262901022740048j)
        25'b000_0000011011100111000000, // (0.3070627100951053+0.02696250364705963j)
        25'b111_1110101110010100000111, // (-0.4730629191512655-0.07977137762215616j)
        25'b000_0111100111011011001110 // (0.267808722530607+0.4760014734738304j)
    };
    """
    r_exp = [(0.2681706279533744 + 0.1967872201553877j), (0.38423295053796597 + 0.09832108527416124j),
             (0.5669563654732931 - 0.0262901022740048j), (0.3070627100951053 + 0.02696250364705963j),
             (-0.4730629191512655 - 0.07977137762215616j), (0.267808722530607 + 0.4760014734738304j),
             ]
    r_exp_real = [Bin2sComp(x.real, pd=pd, p=p) for x in r_exp]
    r_exp_imag = [Bin2sComp(x.imag, pd=pd, p=p) for x in r_exp]

    a, b = 0.300922921527581, 0.19737935744311108
    a, b = Bin2sComp(a, pd=pd, p=p), Bin2sComp(b, pd=pd, p=p)

    # [420. 520. 650. 800. 850. 950.] GHz:
    f = [0.0132038236383, 0.016347591171219998, 0.02043448896403,
         0.02515014026342, 0.02672202402988, 0.02986579156281]
    g = [0.024647137458149997, 0.03051550351962, 0.03814437939952,
         0.04694692849172, 0.04988111152245, 0.055749477583909995]

    f = [Bin2sComp(x, pd=pd, p=p) for x in f]
    g = [Bin2sComp(x, pd=pd, p=p) for x in g]

    pi = Bin2sComp(pi64, pd=pd, p=p)
    pi2_inv = Bin2sComp(1 / (2*pi64), pd=pd, p=p)

    # sine consts:
    B = Bin2sComp(4 / pi64, pd=pd, p=p)
    C = Bin2sComp(-4 / (pi64 * pi64), pd=pd, p=p)
    P = Bin2sComp(0.225, pd=pd, p=p)


    def c_mod(s):
        r_int = trunc(s * pi2_inv)

        res = s - (2 * pi * r_int)

        if res > pi:
            res -= 2 * pi

        return res

    def sine(x):
        y = x * (B + C * abs(x))

        res = P * y * (abs(y) - one_as(y)) + y

        return res

    def cose(x):
        x += 0.5 * pi
        x -= (x > pi) * (2 * pi)

        return sine(x)

    def calc_cost(p_):
        amp_error, phi_error = zero(pd=pd, p=p), zero(pd=pd, p=p)
        for i in range(6):
            f0 = f[i] * p_[0]
            f1 = g[i] * p_[1]
            f2 = f[i] * p_[2]

            s0, s1, s2, s3 = f0 + f1 + f2, f1, f2 - f0, f1 - f0 - f2

            s0, s1, s2, s3 = c_mod(s0), c_mod(s1), c_mod(s2), c_mod(s3)

            ss0, ss1, ss2, ss3 = sine(s0), sine(s1), sine(s2), sine(s3)
            cs0, cs1, cs2, cs3 = cose(s0), cose(s1), cose(s2), cose(s3)

            c0 = 2 * a * (b * b - one_as(b))
            c1 = 2 * b
            c2 = 2 * a * (one_as(b) + b * b)
            c3 = 2 * a * a * b
            c4 = a * a
            c5 = b * b - one_as(b)
            c6 = b * b + one_as(b)
            c7 = 4 * a * b

            d0 = ss1 * cs2

            m01_r = c0 * ss1 * ss2  # 2
            m01_i = c1 * ss0 + c2 * d0 + c3 * ss3  # 4

            m11_r = c5 * (c4 * cs3 - cs0)  # 2
            m11_i = c6 * (c4 * ss3 + ss0) + c7 * d0  # 4

            r_mod_enum_r = m01_r * m11_r + m01_i * m11_i
            r_mod_enum_i = m01_i * m11_r - m01_r * m11_i
            r_mod_denum = m11_r * m11_r + m11_i * m11_i

            amp_diff = (r_mod_enum_r - r_exp_real[i] * r_mod_denum)
            phi_diff = (r_mod_enum_i - r_exp_imag[i] * r_mod_denum)

            amp_error += 0.5 * (amp_diff * amp_diff)
            phi_error += 0.5 * (phi_diff * phi_diff)

        loss = amp_error + phi_error

        return loss

    res = calc_cost(point)

    try:
        point[3] = res
    except IndexError:
        return res

if __name__ == '__main__':
    import time
    """
    // model data (r_exp) for p_sol = [168. 609.  98.], 
    // p = [239.777814149857 476.259423971176 235.382882833481] 
    // => f(p_sol, p) = 8.00341041043292 / 2 = 4.00170520521646 (python) 
    """
    pd, p = 12, 40

    p_test = [239.777814149857, 476.259423971176, 235.382882833481]
    p_test = [Bin2sComp(x, pd = pd, p = p) for x in p_test]
    print(p_test)
    start = time.process_time()
    loss = cost(p_test, pd = pd, p = p)
    print("Runtime: ", time.process_time() - start, "(s)")
    print(loss)

