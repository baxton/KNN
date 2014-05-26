import sys
import os

import numpy as np
import scipy as sp


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

        leg_len = 40000
        
        first_rect = self.create_first_rect(leg_len, rectangles)


        return rectangles
    # end start_logic

    def leg_add_offset(self, leg, x_offset, y_offset):
        for r in leg:
            r.x += x_offset
            r.y += y_offset

    def prepare_rect(self, idx, direction, what_array, x, y):
        r = RectanglesAndHoles.Rect(self.AB[AR_A][idx], self.AB[AR_B][idx])
        if AR_A == what_array:
            r.o = O_AX if D_HORZ == direction else O_AY
        else:
            r.o = O_AY if D_HORZ == direction else O_AX
        if D_VERT == direction:
            r.x = x
            r.y = y
            y += r.h()
        else:
            r.x = x
            r.y = y
            x += r.w()
        r.i = idx
        return r, x, y


    def get_leg(self, leg_len, direction, rectangles):
        leg = []
        t_len = 0

        x = 1000000 - leg_len - 100
        y = 1000000 - leg_len - 100

        while t_len < leg_len:
            d = leg_len - t_len

            la = RectanglesAndHoles.get_best_fit_gt(self.AB[AR_A], self.N, d, rectangles)
            lb = RectanglesAndHoles.get_best_fit_gt(self.AB[AR_B], self.N, d, rectangles)

            if None != la and self.AB[AR_A][la] == d:
                r, x, y = self.prepare_rect(la, direction, AR_A, x, y)
                t_len += r.a
                leg.append(r)
                rectangles[la] = r
            elif None != lb and self.AB[AR_B][lb] == d:
                r, x, y = self.prepare_rect(lb, direction, AR_B, x, y)
                t_len += r.b
                leg.append(r)
                rectangles[lb] = r
            else:
                if None != la and None != lb:
                    if self.AB[AR_A][la] > d and self.AB[AR_B][lb] > d:
                        if self.AB[AR_A][la] > self.AB[AR_B][lb]:
                            r, x, y = self.prepare_rect(lb, direction, AR_B, x, y)
                            t_len += r.b
                            leg.append(r)
                            rectangles[lb] = r                           
                        else:
                            r, x, y = self.prepare_rect(la, direction, AR_A, x, y)
                            t_len += r.a
                            leg.append(r)
                            rectangles[la] = r
                    elif self.AB[AR_A][la] > d:
                        r, x, y = self.prepare_rect(la, direction, AR_A, x, y)
                        t_len += r.a
                        leg.append(r)
                        rectangles[la] = r
                    elif self.AB[AR_B][lb] > d:
                        r, x, y = self.prepare_rect(lb, direction, AR_B, x, y)
                        t_len += r.b
                        leg.append(r)
                        rectangles[lb] = r
                    else:
                        if self.AB[AR_A][la] < self.AB[AR_B][lb]:
                            r, x, y = self.prepare_rect(lb, direction, AR_B, x, y)
                            t_len += r.b
                            leg.append(r)
                            rectangles[lb] = r
                        else:
                            r, x, y = self.prepare_rect(la, direction, AR_A, x, y)
                            t_len += r.a
                            leg.append(r)
                            rectangles[la] = r
                elif None != la:
                    r, x, y = self.prepare_rect(la, direction, AR_A, x, y)
                    t_len += r.a
                    leg.append(r)
                    rectangles[la] = r
                elif None != lb:
                    r, x, y = prepare_rect(lb, direction, AR_B, x, y)
                    t_len += r.b 
                    leg.append(r)
                    rectangles[lb] = r
                else:
                    break
        return leg, t_len
    # end get_leg

    def put_2(self, l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles, stop):
        # define l_bottom's position
        num = len(l_top)
        top_y = l_top[num-1].y # + l_top[num-1].h() / 2
        bottom_y = top_y - l_right_len

        dx = self.border_right - l_bottom[0].x
        dy = bottom_y - l_bottom[0].y

        self.leg_add_offset(l_bottom, dx, dy)
        while not self.has_intersections(l_bottom[0], rectangles):
            self.leg_add_offset(l_bottom, -1, 1);
            bottom_y += 1
        self.leg_add_offset(l_bottom, 1, -1);
        bottom_y -= 1

        # define l_right's position
        bottom_num = len(l_bottom)
        right_num = len(l_right)
        
        top_right = l_top[num-1].x + l_top[num-1].w()
        bottom_right = l_bottom[bottom_num-1].x + l_bottom[bottom_num-1].w()

        if top_right > bottom_right:
            dx = bottom_right - l_right[0].x
            dy = l_bottom[l_bottom_len-1].y - l_right[0].y
            
            self.leg_add_offset(l_right, dx, dy)
            while not self.has_intersections(l_right[right_num-1], rectangles):
                self.leg_add_offset(l_right, 0, 1)
            self.leg_add_offset(l_right, 0, -1)
        elif top_right < bottom_right:
            dx = top_right - l_right[0].x + (1000 if stop else 0)
            dy = l_bottom[bottom_num-1].y + l_right[0].h()

            self.leg_add_offset(l_right, dx, dy)
            zz = 0
            while not self.has_intersections(l_right[0], rectangles):
                self.leg_add_offset(l_right, 0, -1)
                zz += 1
                if zz > 1000: break
            self.leg_add_offset(l_right, 0, 1)
        else:
            dx = top_right - l_right[0].x
            dy = l_bottom[bottom_num-1].x + l_right[0].h()
            self.leg_add_offset(l_right, dx, dy)

        self.update_borders(l_top)
        self.update_borders(l_right)
        self.update_borders(l_bottom)
        
        # end put_2

    def put_3(self, l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles,stop):
        # define l_top's position
        dx = self.border_right - l_top[0].x
        dy = self.border_top - l_top[0].y 

        self.leg_add_offset(l_top, dx, dy)

        while not self.has_intersections(l_top[0], rectangles):
            self.leg_add_offset(l_top, -1, -1)
        self.leg_add_offset(l_top, 1, 1)

        self.put_2(l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles, stop)

        # end put_3
        

    def build_right(self, layer_num, leg_len, rectangles, stop):
        rects_num = layer_num + 1 if layer_num > 0 else 1

        for i in range(rects_num):
            if i == 0:
                # here I need 3 legs
                l_top, l_top_len = self.get_leg(leg_len, D_HORZ, rectangles)
                l_bottom, l_bottom_len = self.get_leg(leg_len, D_HORZ, rectangles)
                l_right, l_right_len = self.get_leg(leg_len, D_VERT, rectangles)    

                self.put_3(l_top, l_top_len, l_right, l_right_len, l_bottom, l_bottom_len, rectangles, stop)

            else:
                # here I need 2 legs
                pass


    def create_first_rect(self, leg_len, rectangles):
        x = 0
        y = 0

        l1, l1_len = self.get_leg(leg_len, D_VERT, rectangles)
        dx = 0 - l1[0].x
        dy = 0 - l1[0].y
        self.leg_add_offset(l1, dx, dy)
        self.update_borders(l1)
        
        self.build_right(0, leg_len, rectangles, False)
        leg_len -= 100
        self.build_right(0, leg_len, rectangles, True)











def main():
    c = RectanglesAndHoles()
    c.process()


main()
