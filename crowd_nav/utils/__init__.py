# from math import asin, pi, cos, sqrt, pow, sin, tan, fabs
#
#
# def get_time(v0, vt, x, y, Oo, Ot):
#     vr = sqrt(pow(v0, 2) + pow(vt, 2) - 2 * v0 * vt * cos(Ot - Oo))
#     if Oo!=Ot:
#         Or = pi - asin(vt / vr * sin(Ot - Oo))
#         O = Oo + Or
#         k = tan(O)
#         DCPA = fabs(y - k * x) / sqrt(1 + k * k)
#         TP = sqrt(x * x + y * y - pow(DCPA, 2))
#         TPCA = TP / vr
#         return TPCA
#     else:
#         pass
#
#
# t = get_time(1, 1, 3, 3, pi / 4,  -3*pi / 4)
# print t

