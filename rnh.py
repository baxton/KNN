import sys
import os

import numpy as np
import scipy as sp


k1 = 1.
k2 = 0.
k3 = 0.
k4 = 0.
k5 = 0.

LEG_LEN = 2000

O_AX=0
O_AY=1

A_TL = 1
A_TR = 2
A_RB = 4
A_BL = 8

AR_A = 0
AR_B = 1

D_VERT = 0
D_HORZ = 1

LT_TOP = 1
LT_RIGHT = 2
LT_BOTTOM = 4
LT_LEFT = 8



class RectanglesAndHoles(object):

    class Rect(object):
        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.x = 0
            self.y = 0
            self.o = O_AX
            self.i = 0

        def w(self):
            return self.a if O_AX == self.o else self.b

        def h(self):
            return self.b if O_AX == self.o else self.a

        def intersect(self, r):
            return not ((self.x + self.w() <= r.x) or (r.x + r.w() <= self.x) or (self.y + self.h() <= r.y) or (r.y + r.h() <= self.y))

    # end Rect

    def __init__(self):
        self.AB = [None, None]
        self.N = 0

        self.border_left = 0
        self.border_top = 0
        self.border_right = 0
        self.border_bottom = 0


    def has_intersections(self, rect, rectangles):
        #res = [ 1 for r in rectangles if None != r and r.i != rect.i and rect.intersect(r) ]
        #return False if not res else True
        for r in rectangles:
            if None != r and r.i != rect.i and rect.intersect(r):
                return True
        return False


    def update_borders(self, r):
        if type(r) is RectanglesAndHoles.Rect:
            if r.x < self.border_left:
                self.border_left = r.x
            if r.x + r.w() > self.border_right:
                self.border_right = r.x + r.w()
            if r.y < self.border_bottom:
                self.border_bottom = r.y
            if r.y + r.h() > self.border_top:
                self.border_top = r.y + r.h()
        else:
            for o in r:
                self.update_borders(o)


    def process(self):
        N, A, B = RectanglesAndHoles.read_data()
        result = self.place(A, B)
        for r in result:
            print r

    @staticmethod
    def read_data():
        N = int(sys.stdin.readline())
        A = [0]*N
        B = [0]*N
        for i in range(N):
            A[i] = int(sys.stdin.readline())
        for i in range(N):
            B[i] = int(sys.stdin.readline())
        return N, sp.array(A), sp.array(B)

    def cmp_rect_idx(a, b):
        return a.i - b.i



    @staticmethod
    def get_best_fit_gt(v, n, length, rectangles):
        best = None
        best_d = -9999999
        for i in range(n):
            if None != rectangles[i]:
                continue
            d = v[i] - length
            if (0 > d and d > best_d) or (0 < d and 0 > best_d) or (0 < d and d < best_d):
                best = i
                best_d = d
            elif d == 0:
                best = i
                break
        return best


    def get_best_fit_gt2(self, length, rectangles):
        best = None
        best_d = -9999999
        best_array = None
        best_array_type = AR_A
        for i in range(self.N):
            if None != rectangles[i]:
                continue

            array = None
            array_type = AR_A

            if self.AB[AR_A][i] > self.AB[AR_B][i]:
                array = self.AB[AR_A]
                array_type = AR_A
            else:
                array = self.AB[AR_B]
                array_type = AR_B

            d = array[i] - length
            if (0 > d and d > best_d) or (0 < d and 0 > best_d) or (0 < d and d < best_d):
                best = i
                best_d = d
                best_array = array
                best_array_type = array_type
            elif d == 0:
                best = i
                best_array = array
                best_array_type = array_type
                break
        return best, best_array, best_array_type


    def place(self, A, B):
        self.N = len(A)
        self.AB[AR_A] = A
        self.AB[AR_B] = B

        full_result = [None] * (self.N * 3)

        #---------------------------------------------------------------

        rectangles = self.start_logic()
        #rectangles.sort(cmp=cmp_rect_idx)

        x = 0
        y = -5000
        for i in range(self.N):
            if None != rectangles[i]:
                full_result[i*3] = rectangles[i].x
                full_result[i*3+1] = rectangles[i].y
                full_result[i*3+2] = rectangles[i].o
            else:
                full_result[i*3] = x
                full_result[i*3+1] = y
                full_result[i*3+2] = O_AX
                x += self.AB[AR_A][i]


        #---------------------------------------------------------------

        return full_result
    # end place

    def start_logic(self):
        rectangles = [None] * self.N

        leg_len = LEG_LEN

        first_rect = self.create_first_rect(leg_len, rectangles)


        return rectangles
    # end start_logic

    def leg_add_offset(self, leg, x_offset, y_offset):
        for r in leg:
            r.x += x_offset
            r.y += y_offset

    def prepare_rect(self, idx, direction, what_array, x, y, leg_type):
        r = RectanglesAndHoles.Rect(self.AB[AR_A][idx], self.AB[AR_B][idx])
        if AR_A == what_array:
            r.o = O_AX if D_HORZ == direction else O_AY
        else:
            r.o = O_AY if D_HORZ == direction else O_AX

        dx = 0 if leg_type == LT_TOP or leg_type == LT_BOTTOM else \
             r.w() if leg_type == LT_LEFT else 0
        dy = 0 if leg_type == LT_LEFT or leg_type == LT_RIGHT else \
             r.h() if leg_type == LT_BOTTOM else 0

        if D_VERT == direction:
            r.x = x - dx
            r.y = y - dy
            y += r.h()
        else:
            r.x = x - dx
            r.y = y - dy
            x += r.w()
        r.i = idx
        return r, x, y


    def get_leg2(self, leg_len, direction, rectangles, leg_type):
        leg = []
        t_len = 0

        x = 1000000 - leg_len - 100
        y = 1000000 - leg_len - 100

        while t_len < leg_len:
            d = leg_len - t_len

            la = RectanglesAndHoles.get_best_fit_gt(self.AB[AR_A], self.N, d, rectangles)
            lb = RectanglesAndHoles.get_best_fit_gt(self.AB[AR_B], self.N, d, rectangles)

            if None != la and self.AB[AR_A][la] == d:
                r, x, y = self.prepare_rect(la, direction, AR_A, x, y, leg_type)
                t_len += r.a
                leg.append(r)
                rectangles[la] = r
            elif None != lb and self.AB[AR_B][lb] == d:
                r, x, y = self.prepare_rect(lb, direction, AR_B, x, y, leg_type)
                t_len += r.b
                leg.append(r)
                rectangles[lb] = r
            else:
                if None != la and None != lb:
                    if self.AB[AR_A][la] > d and self.AB[AR_B][lb] > d:
                        if self.AB[AR_A][la] > self.AB[AR_B][lb]:
                            r, x, y = self.prepare_rect(lb, direction, AR_B, x, y, leg_type)
                            t_len += r.b
                            leg.append(r)
                            rectangles[lb] = r
                        else:
                            r, x, y = self.prepare_rect(la, direction, AR_A, x, y, leg_type)
                            t_len += r.a
                            leg.append(r)
                            rectangles[la] = r
                    elif self.AB[AR_A][la] > d:
                        r, x, y = self.prepare_rect(la, direction, AR_A, x, y, leg_type)
                        t_len += r.a
                        leg.append(r)
                        rectangles[la] = r
                    elif self.AB[AR_B][lb] > d:
                        r, x, y = self.prepare_rect(lb, direction, AR_B, x, y, leg_type)
                        t_len += r.b
                        leg.append(r)
                        rectangles[lb] = r
                    else:
                        if self.AB[AR_A][la] < self.AB[AR_B][lb]:
                            r, x, y = self.prepare_rect(lb, direction, AR_B, x, y, leg_type)
                            t_len += r.b
                            leg.append(r)
                            rectangles[lb] = r
                        else:
                            r, x, y = self.prepare_rect(la, direction, AR_A, x, y, leg_type)
                            t_len += r.a
                            leg.append(r)
                            rectangles[la] = r
                elif None != la:
                    r, x, y = self.prepare_rect(la, direction, AR_A, x, y, leg_type)
                    t_len += r.a
                    leg.append(r)
                    rectangles[la] = r
                elif None != lb:
                    r, x, y = self.prepare_rect(lb, direction, AR_B, x, y, leg_type)
                    t_len += r.b
                    leg.append(r)
                    rectangles[lb] = r
                else:
                    break
        return leg, t_len
    # end get_leg


    def get_leg(self, leg_len, direction, rectangles, leg_type):
        leg = []
        t_len = 0

        x = 1000000 - leg_len - 100
        y = 1000000 - leg_len - 100

        while t_len < leg_len:
            d = leg_len - t_len

            idx, array, array_type = self.get_best_fit_gt2(d, rectangles)

            if None != idx:
                r, x, y = self.prepare_rect(idx, direction, array_type, x, y, leg_type)
                t_len += array[idx]
                leg.append(r)
                rectangles[idx] = r
            else:
                break

        return leg, t_len
    # end get_leg


    def get_borders(self, leg):
        left = leg[0].x
        bottom = leg[0].y
        top = leg[0].y + leg[0].h()
        right = leg[0].x + leg[0].w()

        for r in range(1, len(leg)):
            if leg[r].x < left:
                left = leg[r].x
            if leg[r].y < bottom:
                bottom = leg[r].y
            if leg[r].y + leg[r].h() > top:
                top = leg[r].y + leg[r].h()
            if leg[r].x + leg[r].w() > right:
                right = leg[r].x + leg[r].w()
        return top, right, bottom, left



    def put_2(self, l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles):
        # define l_bottom's position
        top_num = len(l_top)
        bottom_num = len(l_bottom)

        top_t, top_r, top_b, top_l = self.get_borders(l_top)

        target_len = top_b - self.border_bottom
        target_len = l_right_len if target_len > l_right_len else target_len

        dx = self.border_right - l_bottom[0].x
        dy = top_b - target_len - l_bottom[0].h() - l_bottom[0].y
        self.leg_add_offset(l_bottom, dx, dy)

        while not self.has_intersections(l_bottom[0], rectangles):
            self.leg_add_offset(l_bottom, -1, 1);
        while self.has_intersections(l_bottom[0], rectangles):
            self.leg_add_offset(l_bottom, 1, -1);

        # define l_right's position
        bottom_t, bottom_r, bottom_b, bottom_l = self.get_borders(l_bottom)
        #right_t, right_r, right_b, right_l = self.get_borders(l_right)

        target_delta = l_right_len - (top_b - bottom_t)

        #print "1: %d, %d, %d, %d" % (top_b, l_right_len, bottom_b, target_delta)

        right_num = len(l_right)

##        dx = bottom_right + 1000 - l_right[0].x
##        dy = bottom_y - l_right[0].y
##        self.leg_add_offset(l_right, dx, dy)

        if top_r > bottom_r:
            dx = bottom_r - l_right[0].x
            dy = bottom_b - l_right[0].y - target_delta
            self.leg_add_offset(l_right, dx, dy)

##            self.leg_add_offset(l_right, dx, dy)
##            while self.has_intersections(l_right[right_num-1], rectangles):
##                self.leg_add_offset(l_right, 0, -1)
##            get_up = False
##            while not self.has_intersections(l_right[right_num-1], rectangles):
##                get_up = True
##                self.leg_add_offset(l_right, 0, 1)
##            if get_up:
##                self.leg_add_offset(l_right, 0, -1)
        elif top_r < bottom_r:
            dx = top_r - l_right[0].x
            dy = bottom_t - l_right[0].y + target_delta
            self.leg_add_offset(l_right, dx, dy)

            #print "2: %d, %d, %d, %d, %d, %d" % (top_b, l_right_len, bottom_b, bottom_t, target_delta, l_right[0].y)

            while not self.has_intersections(l_right[0], rectangles):
                self.leg_add_offset(l_right, 0, -1)
            while self.has_intersections(l_right[0], rectangles):
                self.leg_add_offset(l_right, 0, 1)
            #print "2: %d, %d, %d, %d, %d, %d" % (top_b, l_right_len, bottom_b, bottom_t, target_delta, l_right[0].y)
        else:
            dx = bottom_r - l_right[0].x
            dy = bottom_b - l_right[0].y - target_delta
            self.leg_add_offset(l_right, dx, dy)

        self.update_borders(l_top)
        self.update_borders(l_right)
        self.update_borders(l_bottom)

        # end put_2

    def put_3(self, l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles):
        # define l_top's position
        dx = self.border_right - l_top[0].x
        dy = self.border_top - l_top[0].y
        self.leg_add_offset(l_top, dx, dy)

        while not self.has_intersections(l_top[0], rectangles):
            self.leg_add_offset(l_top, -1, -1)
        self.leg_add_offset(l_top, 1, 1)

        self.put_2(l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles)

        # end put_3


    def release_leg(self, leg, rectangles):
        for r in leg:
            rectangles[r.i] = None


    def build_right(self, layer_num, leg_len, rectangles):
        rects_num = layer_num + 1 if layer_num > 0 else 1

        l_top, l_top_len = None, 0
        l_bottom, l_bottom_len = None, 0
        l_right, l_right_len = None, leg_len

        for i in range(rects_num):
            if i == 0:
                # here I need 3 legs
                l_top, l_top_len = self.get_leg(leg_len, D_HORZ, rectangles, LT_TOP)
                if not l_top:
                    return False
                l_bottom, l_bottom_len = self.get_leg(leg_len, D_HORZ, rectangles, LT_BOTTOM)
                if not l_bottom:
                    self.release_leg(l_top, rectangles)
                    return False
                l_right, l_right_len = self.get_leg(leg_len, D_VERT, rectangles, LT_RIGHT)
                if not l_right:
                    self.release_leg(l_top, rectangles)
                    self.release_leg(l_bottom, rectangles)
                    return False

                self.put_3(l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles)

            else:
                # here I need 2 legs
                pass
        return True

    def get_nums_available(self, rectangles):
        max_a = 0.
        max_b = 0.
        cnt = 0.
        for r in range(self.N):
            if None != rectangles[r]:
                max_a += self.AB[AR_A][r]
                max_b += self.AB[AR_B][r]
                cnt += 1.
        return cnt, max_a, max_a / cnt, max_b, max_b / cnt

    def create_first_rect(self, leg_len, rectangles):
        x = 0
        y = 0

        # 25, 5000, 200 < 3000
        # 25, 4500, 100 < 2500
        # 100, 2500, 1, 500

        CNT = 300
        ma = self.AB[AR_A].mean()
        mb = self.AB[AR_B].mean()
        m = (ma + mb) / 2
        leg_len = int((self.N * m) / 4 / CNT)
        leg_len = 1024


        l1, l1_len = self.get_leg(leg_len, D_VERT, rectangles, LT_LEFT)
        dx = 0 - l1[0].x
        dy = 0 - l1[0].y
        self.leg_add_offset(l1, dx, dy)
        self.update_borders(l1)

        # 1.5
        # 2.
        # 2.5
        # 2.6

        k = 1.

        for z in range(CNT):
            cnt, max_a, avr_a, max_b, avr_b = self.get_nums_available(rectangles)
            #m = (avr_a + avr_b) / 2.
            leg_len = avr_a * .702 + avr_b * .6905
            leg_len = int(leg_len)
##            if leg_len <= 0:
##                leg_len = 100
            res = self.build_right(0, leg_len, rectangles)
            if not res:
                leg_len /= 4
                cnt = 0
                for zz in range(5):
                    res = self.build_right(0, leg_len, rectangles)
                    if not res:
                        cnt += 1
                        leg_len /= 2
                    else:
                        cnt = 0
                    if cnt == 2:
                        z = CNT
                        break











def main():
    c = RectanglesAndHoles()
    c.process()



if __name__ == '__main__':
    main()
