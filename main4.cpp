

//
// g++ -g  -mmmx     -msse -msse2 -msse3  -D DEBUG -I. -I/c/Working/MinGW/lib/gcc/mingw32/4.8.1/include  main3.cpp -o composer.exe
// g++ --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3  -D DEBUG -I. -I/c/Working/MinGW/lib/gcc/mingw32/4.8.1/include  main4.cpp -o composer.exe
//
// java -jar CollageMakerVis.jar -exec "<command>"
//


#include <cstring>
#include <vector>
#include <cmath>
#include <iterator>
#include <deque>
#include <algorithm>
#include <memory>
#include <pmmintrin.h>
#include <time.h>
#include <map>

#if defined DEBUG
  #include <iostream>
  #include <fstream>

  std::fstream test_log("prep_c.log", std::fstream::out);
#endif


#if defined DEBUG
  #define DBUG1(x1) {test_log << x1 << std::endl;}
  #define DBUG2(x1, x2) {test_log << x1 << x2 << std::endl;}
  #define DBUG_ARR(v) {for (int i = 0; i < v.size(); ++i) test_log << v[i] << ","; test_log << std::endl;}
  #define DBUG_ARR_STATIC(v, size) {for (int i = 0; i < size; ++i) test_log << v[i] << ","; test_log << std::endl;}
#else
  #define DBUG1(x1)
  #define DBUG2(x1, x2)
  #define DBUG_ARR(v)
  #define DBUG_ARR_STATIC(v, size)
#endif


void copy(void* dst, void* src, size_t size) {
    // size is in bytes
    // size is a multiple of 128 (32 byte allignment and pointer to float)
    float* pin = (float*)src;
    float* pout = (float*)dst;

    size_t iterations = size >> 7; // i.e. devide by 32 and by 4
    for (size_t i = 0; i < iterations; ++i) {
        __m128 z1 = _mm_load_ps(pin + 0);
        __m128 z2 = _mm_load_ps(pin + 4);
        __m128 z3 = _mm_load_ps(pin + 8);
        __m128 z4 = _mm_load_ps(pin + 12);
        __m128 z5 = _mm_load_ps(pin + 16);
        __m128 z6 = _mm_load_ps(pin + 20);
        __m128 z7 = _mm_load_ps(pin + 24);
        __m128 z8 = _mm_load_ps(pin + 28);
        _mm_store_ps(pout + 0, z1);
        _mm_store_ps(pout + 4, z2);
        _mm_store_ps(pout + 8, z3);
        _mm_store_ps(pout + 12, z4);
        _mm_store_ps(pout + 16, z5);
        _mm_store_ps(pout + 20, z6);
        _mm_store_ps(pout + 24, z7);
        _mm_store_ps(pout + 28, z8);
        pin += 32;
        pout += 32;
    }
}



using namespace std;


template<class F>
class BeamFit {

    struct Ray {
    };

public:
    void best_fits(vector<F>& fits) {
    }
};



template<class I, class C>
class Beam {
    enum {
        MAX_CELLS = 20*20,
        MAX_IMAGES = 200,
        BEAM_WIDTH = 50,
    };

    struct Ray {
        int cells [MAX_CELLS];
        int images [MAX_IMAGES];
        double total_score;

        Ray() : total_score(0.) {
            for (int i = 0; i < MAX_CELLS; ++i)
                cells[i] = -1;
            for (int i = 0; i < MAX_IMAGES; ++i)
                images[i] = 0;
        }
    };


    int cur_cell_idx;
    int beam_width;
    int cur_beam;
    vector<Ray> beam[2];

public:
    Beam() :
        cur_cell_idx(0),
        beam_width(BEAM_WIDTH),
        cur_beam(0)
    {
        beam[0].reserve(beam_width);
        beam[1].reserve(beam_width);
    }

    void replace(int beam_idx, Ray& r) {
        int min_idx = 0;
        double min_score = beam[beam_idx][min_idx].total_score;

        for (int i = 1; i < beam[beam_idx].size(); ++i) {
            if (min_score > beam[beam_idx][i].total_score) {
                min_idx = i;
                min_score = beam[beam_idx][min_idx].total_score;
            }
        }

        if (min_score < r.total_score) {
            beam[beam_idx][min_idx] = r;
        }
    }

    void start(vector<I>& images, vector<C>& cells) {
        if (1 == cells[cur_cell_idx].skeep) {
                Ray r;
                r.total_score += cells[cur_cell_idx].score;
                r.cells[cur_cell_idx] = cells[cur_cell_idx].img_num;
                r.images[cells[cur_cell_idx].img_num] = 1;

                beam[cur_beam].push_back(r);
        }
        else {
            int n = 0;

            for (int i = 0; i < images.size(); ++i) {
                I& img = images[i];

                if (1 == img.skeep)
                    continue;

                Ray r;
                r.total_score += img.cells[cur_cell_idx];
                r.cells[cur_cell_idx] = i;
                r.images[i] = 1;

                if (n < beam_width) {
                    beam[cur_beam].push_back(r);
                }
                else {
                    replace(cur_beam, r);
                }
                ++n;
            }
        }
        ++cur_cell_idx;
        cur_beam = (cur_beam + 1) % 2;
    }

    void step(vector<I>& images, vector<C>& cells) {
        int n = 0;
        int prev_beam = (cur_beam + 1) % 2;

        if (1 == cells[cur_cell_idx].skeep) {
            for (int r = 0; r < beam[prev_beam].size(); ++r) {
                Ray& ray = beam[prev_beam][r];

                Ray child = ray;
                child.total_score += cells[cur_cell_idx].score;
                child.cells[cur_cell_idx] = cells[cur_cell_idx].img_num;
                child.images[cells[cur_cell_idx].img_num] = 1;

                beam[cur_beam].push_back(child);
            }
        }
        else {
            for (int r = 0; r < beam[prev_beam].size(); ++r) {
                Ray& ray = beam[prev_beam][r];

                int skeepped = 0;
                for (int i = 0; i < images.size(); ++i) {
                    I& img = images[i];
                    if (0 == img.skeep && ray.images[i] == 0) {

                        Ray child = ray;
                        child.total_score += img.cells[cur_cell_idx];
                        child.cells[cur_cell_idx] = i;
                        child.images[i] = 1;

                        if (n < beam_width) {
                            beam[cur_beam].push_back(child);
                        }
                        else {
                            replace(cur_beam, child);
                        }
                        ++n;
                    }
                    else {
                        ++skeepped;
                    }
                }
            }
        }

        ++cur_cell_idx;
        beam[prev_beam].clear();
        cur_beam = (cur_beam + 1) % 2;
    }

    void best_layout(vector<I>& images, vector<C>& cells, vector<int>& result) {
        beam[0].clear();
        beam[1].clear();

        int cells_num = cells.size();

        DBUG2("Cells: ", cells_num)

        DBUG1("Start beam")
        start(images, cells);

        DBUG1("Step beam")
        while (cur_cell_idx < cells_num) {
            step(images, cells);
        }

        int beam_idx = (cur_beam + 1) % 2;

        int max_idx = 0;
        double max_score = beam[beam_idx][max_idx].total_score;

        for (int r = 1; r < beam[beam_idx].size(); ++r) {
            Ray& ray = beam[beam_idx][r];

            if (max_score < ray.total_score) {
                max_idx = r;
                max_score = beam[beam_idx][max_idx].total_score;
            }
        }

        DBUG2("Best ray #", max_idx)

        Ray& best_ray = beam[beam_idx][max_idx];
        for (int i = 0; i < cells_num; ++i) {
            DBUG2(" cell #", i)
            DBUG2(" img #", best_ray.cells[i])
            result.push_back(best_ray.cells[i]);
        }
    }
};



#define CELL_H 100
#define CELL_W 100

int new_img_1[CELL_H * CELL_W] __attribute__ ((aligned (16)));
int new_img_2[CELL_H * CELL_W] __attribute__ ((aligned (16)));
int new_img_3[CELL_H * CELL_W] __attribute__ ((aligned (16)));
int new_img_4[CELL_H * CELL_W] __attribute__ ((aligned (16)));




class CollageMaker {


    // 62 99 143.89
    // 50 100
    int GRID_H;
    int GRID_W;


    enum {
        //NUMBER_GRID = 5,    // min 5 - 13

        TARGET_H = 400,
        TARGET_W = 400,

        //CELL_H = 100,
        //CELL_W = 100,

        MAX_CELLS = 20*20,

        NO_CELL = -1,
        NO_IMG = -1,
    };

    int target [TARGET_H * TARGET_W] __attribute__ ((aligned (16)));
    int target_h;
    int target_w;


    struct Cell {
        int idx;
        int top_row;
        int left_col;
        int bottom_row;
        int right_col;
        int h;
        int w;
        int img_offset;
        int img_num;
        int skeep;
        int used;
        double score;
    } __attribute__ ((aligned (16)));


    struct Img {
        int idx;
        int offset;
        int h;
        int w;
        int used_in;
        int skeep;
        double cells[MAX_CELLS] __attribute__ ((aligned (16)));
    } __attribute__ ((aligned (16)));


    void copy(int* __restrict__ dst, int* __restrict__ src, int size) {
        for (int i = 0; i < size; ++i)
            dst[i] = src[i];
    }

    bool double_eq(double d1, double d2) {
        double e = 0.001;
        double diff = d1 > d2 ? (d1 - d2) : (d2 - d1);
        return diff <= e;
    }

    void scale(vector<int>& data, int offset, int H, int W, int new_h, int new_w, int* __restrict__ new_img) {
        int size = new_h * new_w;
        for (int i = 0; i < size; ++i) {
            new_img[i] = 0;
        }

        double rowK = H / (double)new_h;
        double colK = W / (double)new_w;

        int new_r = 0;
        int new_c = 0;

        double rev_rm = 0.;
        double rev_cm = 0.;

        double trk = rowK;
        for (int r = 0; r < H; ++r) {
            double rm = 0.;
            if (trk >= 1.) {
                rm = 1.;
                trk -= 1.;
            }
            else {
                rm = trk;
                trk = 0.;
            }

            rev_rm = 0.;
            if (rm < 1.) {
                rev_rm = 1. - rm;
            }

            double tck = colK;
            for (int c = 0; c < W; ++c) {
                double cm = 0.;
                if (tck >= 1.) {
                    cm = 1.;
                    tck -= 1.;
                }
                else {
                    cm = tck;
                    tck = 0.;
                }

                int orig_idx = r * W + c;
                int new_idx = new_r * new_w + new_c;

                double p = data[offset + orig_idx] * (rm * cm);

                new_img[new_idx] += p;

                // check next point
                rev_cm = 0.;
                if (cm < 1.) {
                    rev_cm = 1. - cm;
                }

                if (rev_rm > 0. && rev_cm == 0.) {
                    new_idx = (new_r + 1) * new_w + new_c;
                    p = data[offset + orig_idx] * rev_rm;
                    new_img[new_idx] += p;
                }
                else if (rev_rm == 0. && rev_cm > 0.) {
                    new_idx = new_r * new_w + (new_c + 1);
                    p = data[offset + orig_idx] * rev_cm;
                    new_img[new_idx] += p;
                }
                else {
                    double d = data[offset + orig_idx] * (rev_rm * rev_cm);
                    double tr = data[offset + orig_idx] * rev_rm - d;
                    double tc = data[offset + orig_idx] * rev_cm - d;

                    new_idx = (new_r + 1) * new_w + new_c;
                    new_img[new_idx] += tr;

                    new_idx = new_r * new_w + (new_c + 1);
                    new_img[new_idx] += tc;

                    new_idx = (new_r + 1) * new_w + (new_c + 1);
                    new_img[new_idx] += d;
                }

                //
                if (double_eq(tck, 0.)) {
                    new_c = (new_c + 1) % new_w;
                    tck = colK;
                    if (!double_eq(rev_cm, 0.))
                        tck -= rev_cm;
                }
            }
            //
            if (double_eq(trk, 0.)) {
                new_r = (new_r + 1) % new_h;
                trk = rowK;
                if (!double_eq(rev_rm, 0.))
                    trk -= rev_rm;
            }
        }
    }

    void downscale(vector<int>& data, int offset, int H, int W, int new_h, int new_w, int* __restrict__ new_img) {
        int size = new_h * new_w;
        for (int i = 0; i < size; ++i) {
            new_img[i] = 0;
        }

        int big_h = H * new_h;
        int big_w = W * new_w;

        vector<int> numbers(size, 0);

        for (int r = 0; r < big_h; ++r) {
            for (int c = 0; c < big_w; ++c) {
                int idx = r/H * new_w + c/W;
                int orig_idx = r/new_h * W + c/new_w;

                new_img[idx] += data[offset + orig_idx];
                numbers[idx] += 1;
            }
        }

        double k = H * W;
        for (int i = 0; i < size; ++i) {
            new_img[i] = (int)round((double)new_img[i] / numbers[i]);
        }
    }

    void get_random_indices(std::vector<int>& indices, int number, int length=200) {
        int n = 0;

        for (int i = 0; i < length; ++i) {
            ++n;
            if (n <= number) {
                indices.push_back(i);
            }
            else {
                int idx = rand() % n;
                if (idx < number) {
                    indices[idx] = i;
                }
            }
        }
    }


    int get_target(vector<int>& data) {
        target_h = data[0];
        target_w = data[1];

        DBUG2("H: ", target_h);
        DBUG2("W: ", target_w);

        int size = target_h * target_w;

        copy(target, &data[2], size);

        return size+2;  // points to the next image
    }

    // this works with non-downscaled images
    void get_score_exact(vector<int>& data,
                     int offset,
                     int top_row,
                     int left_col,
                     double& score) {

        int h = data[offset++];
        int w = data[offset++];

        score = 0.;

        for (int r = top_row; (r < top_row + h) && (r < target_h); ++r) {
            for (int c = left_col; (c < left_col + w) && (c < target_w); ++c) {
                int target_idx = r * target_w + c;
                int img_idx = (r - top_row) * w + (c - left_col);


                int diff = target[target_idx] - data[offset + img_idx];
                score += diff*diff;
            }
        }

        score = sqrt(score / (h * w));
    }

    void get_score_exact2(int* __restrict__ new_img,
                     int h,
                     int w,
                     int top_row,
                     int left_col,
                     double& score) {

        score = 0.;

        for (int r = top_row; (r < top_row + h) && (r < target_h); ++r) {
            for (int c = left_col; (c < left_col + w) && (c < target_w); ++c) {
                int target_idx = r * target_w + c;
                int img_idx = (r - top_row) * w + (c - left_col);


                int diff = target[target_idx] - new_img[img_idx];
                score += diff*diff;
            }
        }

        score = sqrt(score / (h * w));
    }

    struct Fit {
        int img_idx;
        int row;
        int column;
        int scale;
        int h;
        int w;
        double score;
    };


    void get_best_position2(int* __restrict__ new_img, int h, int w, int step, int& row, int& column, double& score,
                            vector<Fit>& fits, int max_fits, int img_num) {
        //double best_score = 0.;
        //int best_top_row = 0;
        //int best_left_col = 0;

        for (int r = 0; r < target_h; r += step) {
            for (int c = 0; c < target_w; c += step)  {
                int cur_row = (r + h <= target_h ? r : target_h - h);
                int cur_col = (c + w <= target_w ? c : target_w - w);

                double score = 0.;
                get_score_exact2(new_img, h, w, cur_row, cur_col, score);

                //if (best_score < score) {
                //    best_score = score;
                //    best_top_row = cur_row;
                //    best_left_col = cur_col;
                //}
                Fit f;
                f.img_idx = img_num;
                f.row = cur_row;
                f.column = cur_col;
                f.score = score;
                f.h = h;
                f.w = w;

                if (fits.size() < max_fits)
                    fits.push_back(f);
                else
                    replace_fit(fits, f);
            }
        }

        //score = best_score;
        //row = best_top_row;
        //column = best_left_col;
    }

    void get_best_position(vector<int>& data, int offset, int step, int& row, int& column, double& score,
                           vector<Fit>& fits, int max_fits, int img_num) {
        int h = data[offset+0];
        int w = data[offset+1];

        //double best_score = 0.;
        //int best_top_row = 0;
        //int best_left_col = 0;

        for (int r = 0; r < target_h; r += step) {
            for (int c = 0; c < target_w; c += step)  {
                int cur_row = (r + h <= target_h ? r : target_h - h);
                int cur_col = (c + w <= target_w ? c : target_w - w);

                double score = 0.;
                get_score_exact(data, offset, cur_row, cur_col, score);

                //if (best_score < score) {
                //    best_score = score;
                //    best_top_row = cur_row;
                //    best_left_col = cur_col;
                //}
                Fit f;
                f.img_idx = img_num;
                f.row = cur_row;
                f.column = cur_col;
                f.score = score;
                f.h = h;
                f.w = w;

                if (fits.size() < max_fits)
                    fits.push_back(f);
                else
                    replace_fit(fits, f);
            }
        }

        //score = best_score;
        //row = best_top_row;
        //column = best_left_col;
    }

    void prepare_matrix(vector<Cell>& matrix) {
        int img_num = 0;

        for (int r = 0; r < target_h; r += GRID_H) {
            for (int c = 0; c < target_w; c += GRID_W) {
                Cell cell;

                // initializing
                cell.img_offset = 0;
                cell.img_num = -1;
                cell.score = 0.0;
                cell.used = 1;
                cell.skeep = 0;

                // populating
                cell.idx = img_num++;
                cell.top_row = r;
                cell.bottom_row = cell.top_row + GRID_H - 1;
                cell.left_col = c;
                cell.right_col = cell.left_col + GRID_W - 1;
                cell.h = GRID_H;
                cell.w = GRID_W;

                if (cell.bottom_row > target_h) {
                    cell.bottom_row = target_h - 1;
                    cell.h = cell.bottom_row - cell.top_row + 1;
                }
                if (cell.right_col > target_w) {
                    cell.right_col = target_w - 1;
                    cell.w = cell.right_col - cell.left_col + 1;
                }

                matrix.push_back(cell);
            }
        }
    }

    void prepare_images(vector<int>& data, int& cur_idx, vector<Img>& images) {
        int img_num = 0;
        int size = data.size();

        while (cur_idx < size) {
            Img img;

            img.used_in = -1;
            img.skeep = 0;

            // populate
            img.idx = img_num++;
            img.offset = cur_idx;
            img.h = data[cur_idx++];
            img.w = data[cur_idx++];

            DBUG2("Prep imgs #", img.idx)
            DBUG2("Prep imgs h: ", img.h)
            DBUG2("Prep imgs w: ", img.w)

            //if (img.h < GRID_H)
            //    GRID_H = img.h;
            //if (img.w < GRID_W)
            //    GRID_W = img.w;

            for (int i = 0; i < MAX_CELLS; ++i)
                img.cells[i] = 0.;

            images.push_back(img);

            cur_idx += img.h * img.w;
        }
    }


    void fill_scores_approx(vector<int>& data,
                     int offset,
                     vector<Cell>& cells,
                     Img& img_data) {

        int h = data[offset++];
        int w = data[offset++];
        int max_idx = h * w;

        for (int c = 0; c < cells.size(); ++c) {
            Cell& cell = cells[c];
            if (1 == cell.skeep)
                continue;

            //scale(data, offset, img_data.h, img_data.w, cell.h, cell.w, new_img_1);


            for (int r = cell.top_row; r <= cell.bottom_row; ++r) {
                for (int c = cell.left_col; c <= cell.right_col; ++c) {
                    int kh = h / cell.h;
                    int kw = w / cell.w;

                    int target_idx = r * target_w + c;
                    int img_idx = (r - cell.top_row) * kh * cell.w + (c - cell.left_col) * kw;
                    if (img_idx >= max_idx)
                        img_idx = max_idx - 1;

                    double sum = 0;
                    int cnt = 0.;
                    for (int z = img_idx - 2; z < img_idx + 2; ++z) {
                        if (0 <= z && z < max_idx) {
                            sum += data[offset + z];
                            cnt += 1;
                        }
                    }
                    sum /= cnt;

                    int diff = target[target_idx] - sum;
                    img_data.cells[cell.idx] += diff*diff;
                }
            }

            img_data.cells[cell.idx] = sqrt(img_data.cells[cell.idx] / (cell.h * cell.w));
        }
    }


    void fill_scores_scale(vector<int>& data,
                     int offset,
                     vector<Cell>& cells,
                     Img& img_data) {

        int h = data[offset++];
        int w = data[offset++];
        int max_idx = h * w;

        for (int c = 0; c < cells.size(); ++c) {
            Cell& cell = cells[c];
            if (1 == cell.skeep)
                continue;

            scale(data, offset, img_data.h, img_data.w, cell.h, cell.w, new_img_1);

            for (int r = cell.top_row; r <= cell.bottom_row; ++r) {
                for (int c = cell.left_col; c <= cell.right_col; ++c) {

                    int target_idx = r * target_w + c;
                    int img_idx = (r - cell.top_row) * cell.w + (c - cell.left_col);

                    int diff = target[target_idx] - new_img_1[img_idx];
                    img_data.cells[cell.idx] += diff*diff;
                }
            }

            img_data.cells[cell.idx] = sqrt(img_data.cells[cell.idx] / (cell.h * cell.w));
        }
    }


    struct cmp_fit_desc {
        bool operator() (const Fit& f1, const Fit& f2) {
            return f1.score > f2.score;
        }
    };

    bool intersect(const Cell& c1, const Cell& c2) {
        bool no_intersection = (c1.left_col + c1.w <= c2.left_col) || (c2.left_col + c2.w <= c1.left_col) ||
                               (c1.top_row + c1.h <= c2.top_row) || (c2.top_row + c2.h <= c1.top_row);
        return !no_intersection;
    }

    bool plane_intersect(int r1, int h1, int c1, int w1, int r2, int h2, int c2, int w2) {
        bool no_intersection = (c1 + w1 <= c2) || (c2 + w2 <= c1) ||
                               (r1 + h1 <= r2) || (r2 + h2 <= r1);
        return !no_intersection;
    }

    void print_cell(Cell& cell) {
        DBUG2("Pnt cell #", cell.idx)
        DBUG2("Pnt cell row: ", cell.top_row)
        DBUG2("Pnt cell col: ", cell.left_col)
        DBUG2("Pnt cell h: ", cell.h)
        DBUG2("Pnt cell w: ", cell.w)
    }

    void add_cell(vector<Cell>& matrix, Cell& cell) {
        vector<int> to_delete;
        vector<Cell> new_cells;
    DBUG2("inserting img #", cell.img_num)
        for (int c = 0; c < matrix.size(); ++c) {
            Cell& old = matrix[c];

            if (intersect(old, cell)) {
#if defined DEBUG
    print_cell(old);
#endif
                if (old.top_row >= cell.top_row &&
                    old.left_col >= cell.left_col &&
                    old.right_col <= cell.right_col &&
                    old.bottom_row <= cell.bottom_row) {
                    // equal or old is in cell
                    to_delete.push_back(old.idx);
    DBUG1("deleted")
                }

                else if (old.top_row < cell.top_row && old.bottom_row > cell.bottom_row &&
                         old.left_col < cell.left_col && old.right_col > cell.right_col) {
                    // cell is inside
                    Cell new_cell1 = old;
                    Cell new_cell2 = old;
                    Cell new_cell3 = old;

                    old.bottom_row = cell.top_row - 1;
                    old.h = old.bottom_row - old.top_row + 1;

                    new_cell1.top_row = cell.top_row;
                    new_cell1.right_col = cell.left_col - 1;
                    new_cell1.h = new_cell1.bottom_row - new_cell1.top_row + 1;
                    new_cell1.w = new_cell1.right_col - new_cell1.left_col + 1;

                    new_cell2.top_row = cell.top_row;
                    new_cell2.left_col = cell.right_col + 1;
                    new_cell2.h = new_cell2.bottom_row - new_cell2.top_row + 1;
                    new_cell2.w = new_cell2.right_col - new_cell2.left_col + 1;

                    new_cell3.top_row = cell.bottom_row + 1;
                    new_cell3.right_col = cell.right_col;
                    new_cell3.left_col = cell.left_col;
                    new_cell3.h = new_cell3.bottom_row - new_cell3.top_row + 1;
                    new_cell3.w = new_cell3.right_col - new_cell3.left_col + 1;

                    new_cells.push_back(new_cell1);
                    new_cells.push_back(new_cell2);
                    new_cells.push_back(new_cell3);
#if defined DEBUG
    DBUG1("inside")
    print_cell(old);
    print_cell(new_cell1);
    print_cell(new_cell2);
    print_cell(new_cell3);
#endif
                }

                else if (old.top_row >= cell.top_row && old.left_col < cell.left_col &&
                    old.right_col > cell.right_col && old.bottom_row > cell.bottom_row &&
                    old.top_row < cell.bottom_row) {
                    //
                    Cell new_cell1 = old;
                    Cell new_cell2 = old;

                    old.top_row = cell.bottom_row + 1;
                    old.h = old.bottom_row - old.top_row + 1;

                    new_cell1.right_col = cell.left_col - 1;
                    new_cell1.bottom_row = cell.bottom_row;
                    new_cell1.w = new_cell1.right_col - new_cell1.left_col + 1;
                    new_cell1.h = new_cell1.bottom_row - new_cell1.top_row + 1;

                    new_cell2.left_col = cell.right_col + 1;
                    new_cell2.bottom_row = cell.bottom_row;
                    new_cell2.w = new_cell2.right_col - new_cell2.left_col + 1;
                    new_cell2.h = new_cell2.bottom_row - new_cell2.top_row + 1;

                    new_cells.push_back(new_cell1);
                    new_cells.push_back(new_cell2);
#if defined DEBUG
    DBUG1("from top")
    print_cell(old);
    print_cell(new_cell1);
    print_cell(new_cell2);
#endif
                }

                else if (old.top_row < cell.top_row && old.left_col < cell.left_col &&
                    old.right_col > cell.right_col && old.bottom_row <= cell.bottom_row &&
                    old.bottom_row > cell.top_row) {
                    //
                    Cell new_cell1 = old;
                    Cell new_cell2 = old;

                    old.bottom_row = cell.top_row - 1;
                    old.h = old.bottom_row - old.top_row + 1;

                    new_cell1.right_col = cell.left_col - 1;
                    new_cell1.top_row = cell.top_row;
                    new_cell1.w = new_cell1.right_col - new_cell1.left_col + 1;
                    new_cell1.h = new_cell1.bottom_row - new_cell1.top_row + 1;

                    new_cell2.left_col = cell.right_col + 1;
                    new_cell2.top_row = cell.top_row;
                    new_cell2.w = new_cell2.right_col - new_cell2.left_col + 1;
                    new_cell2.h = new_cell2.bottom_row - new_cell2.top_row + 1;

                    new_cells.push_back(new_cell1);
                    new_cells.push_back(new_cell2);
#if defined DEBUG
    DBUG1("from bottom")
    print_cell(old);
    print_cell(new_cell1);
    print_cell(new_cell2);
#endif
                }

                else if (old.top_row < cell.top_row && old.left_col >= cell.left_col &&
                    old.right_col > cell.right_col && old.bottom_row > cell.bottom_row &&
                    old.left_col < cell.right_col) {
                    //
                    Cell new_cell1 = old;
                    Cell new_cell2 = old;

                    old.left_col = cell.right_col + 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell1.right_col = cell.right_col;
                    new_cell1.bottom_row = cell.top_row - 1;
                    new_cell1.w = new_cell1.right_col - new_cell1.left_col + 1;
                    new_cell1.h = new_cell1.bottom_row - new_cell1.top_row + 1;

                    new_cell2.right_col = cell.right_col;
                    new_cell2.top_row = cell.bottom_row + 1;
                    new_cell2.w = new_cell2.right_col - new_cell2.left_col + 1;
                    new_cell2.h = new_cell2.bottom_row - new_cell2.top_row + 1;

                    new_cells.push_back(new_cell1);
                    new_cells.push_back(new_cell2);
#if defined DEBUG
    DBUG1("from left")
    print_cell(old);
    print_cell(new_cell1);
    print_cell(new_cell2);
#endif
                }

                else if (old.top_row < cell.top_row && old.left_col < cell.left_col &&
                    old.right_col <= cell.right_col && old.bottom_row > cell.bottom_row &&
                    old.right_col > cell.left_col) {
                    //
                    Cell new_cell1 = old;
                    Cell new_cell2 = old;

                    old.right_col = cell.left_col - 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell1.left_col = cell.left_col;
                    new_cell1.bottom_row = cell.top_row - 1;
                    new_cell1.w = new_cell1.right_col - new_cell1.left_col + 1;
                    new_cell1.h = new_cell1.bottom_row - new_cell1.top_row + 1;

                    new_cell2.left_col = cell.left_col;
                    new_cell2.top_row = cell.bottom_row + 1;
                    new_cell2.w = new_cell2.right_col - new_cell2.left_col + 1;
                    new_cell2.h = new_cell2.bottom_row - new_cell2.top_row + 1;

                    new_cells.push_back(new_cell1);
                    new_cells.push_back(new_cell2);
#if defined DEBUG
    DBUG1("from right")
    print_cell(old);
    print_cell(new_cell1);
    print_cell(new_cell2);
#endif
                }

                else if (old.top_row < cell.top_row && old.left_col < cell.left_col) {
                    // cell's top-left corner is in
                    Cell new_cell = old;

                    old.right_col = cell.left_col - 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell.left_col = cell.left_col;
                    new_cell.bottom_row = cell.top_row - 1;
                    new_cell.w = new_cell.right_col - new_cell.left_col + 1;
                    new_cell.h = new_cell.bottom_row - new_cell.top_row + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("top_left")
    print_cell(old);
    print_cell(new_cell);
#endif
                }
                else if (old.top_row < cell.top_row && old.right_col > cell.right_col) {
                    // cell's top-right corner is in
                    Cell new_cell = old;

                    old.left_col = cell.right_col + 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell.right_col = cell.right_col;
                    new_cell.bottom_row = cell.top_row - 1;
                    new_cell.w = new_cell.right_col - new_cell.left_col + 1;
                    new_cell.h = new_cell.bottom_row - new_cell.top_row + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("top_right")
    print_cell(old);
    print_cell(new_cell);
#endif
                }
                else if (old.bottom_row > cell.bottom_row && old.left_col < cell.left_col) {
                    // cell's bottom-left corner is in
                    Cell new_cell = old;

                    old.right_col = cell.left_col - 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell.left_col = cell.left_col;
                    new_cell.top_row = cell.bottom_row + 1;
                    new_cell.w = new_cell.right_col - new_cell.left_col + 1;
                    new_cell.h = new_cell.bottom_row - new_cell.top_row + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("bottom_left")
    print_cell(old);
    print_cell(new_cell);
#endif
                }
                else if (old.right_col > cell.right_col && old.bottom_row > cell.bottom_row) {
                    // cell's bottom-right corner is in
                    Cell new_cell = old;

                    old.left_col = cell.right_col + 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell.right_col = cell.right_col;
                    new_cell.top_row = cell.bottom_row + 1;
                    new_cell.w = new_cell.right_col - new_cell.left_col + 1;
                    new_cell.h = new_cell.bottom_row - new_cell.top_row + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("top_right")
    print_cell(old);
    print_cell(new_cell);
#endif
                }

                else if (old.top_row >= cell.top_row && old.bottom_row <= cell.bottom_row &&
                         old.left_col < cell.left_col && old.right_col > cell.right_col) {
                    // between vert
                    Cell new_cell = old;

                    old.right_col = cell.left_col - 1;
                    old.w = old.right_col - old.left_col + 1;

                    new_cell.left_col = cell.right_col + 1;
                    new_cell.w = new_cell.right_col - new_cell.left_col + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("xross vert")
    print_cell(old);
    print_cell(new_cell);
#endif
                }

                else if (old.left_col >= cell.left_col && old.right_col <= cell.right_col &&
                         old.top_row < cell.top_row && old.bottom_row > cell.bottom_row) {
                    // between horz
                    Cell new_cell = old;

                    old.bottom_row = cell.top_row - 1;
                    old.h = old.bottom_row - old.top_row + 1;

                    new_cell.top_row = cell.bottom_row + 1;
                    new_cell.h = new_cell.bottom_row - new_cell.top_row + 1;

                    new_cells.push_back(new_cell);
#if defined DEBUG
    DBUG1("xross horz")
    print_cell(old);
    print_cell(new_cell);
#endif
                }

                else if (old.top_row < cell.top_row && old.bottom_row <= cell.bottom_row &&
                         old.left_col >= cell.left_col && old.right_col <= cell.right_col) {
                    old.bottom_row = cell.top_row - 1;
                    old.h = old.bottom_row - old.top_row + 1;
#if defined DEBUG
    DBUG1("down")
    print_cell(old);
#endif
                }
                else if (old.left_col < cell.left_col && old.right_col <= cell.right_col &&
                        old.top_row >= cell.top_row && old.bottom_row <= cell.bottom_row) {
                    old.right_col = cell.left_col - 1;
                    old.w = old.right_col - old.left_col + 1;
#if defined DEBUG
    DBUG1("right")
    print_cell(old);
#endif
                }
                else if (old.right_col > cell.right_col && old.left_col >= cell.left_col &&
                         old.top_row >= cell.top_row && old.bottom_row <= cell.bottom_row) {
                    old.left_col = cell.right_col + 1;
                    old.w = old.right_col - old.left_col + 1;
#if defined DEBUG
    DBUG1("left")
    print_cell(old);
#endif
                }
                else if (old.top_row >= cell.top_row && old.bottom_row > cell.bottom_row &&
                         old.left_col >= cell.left_col && old.right_col <= cell.right_col) {
                    old.top_row = cell.bottom_row + 1;
                    old.h = old.bottom_row - old.top_row + 1;
#if defined DEBUG
    DBUG1("up")
    print_cell(old);
#endif
                }
            }
        }

        matrix.push_back(cell);

        for (int i = 0; i < to_delete.size(); ++i)
            for (int d = 0; d < matrix.size(); ++d) {
                if (to_delete[i] == matrix[d].idx) {
                    matrix.erase(matrix.begin() + d);
                    break;
                }
            }

        for (int i = 0; i < new_cells.size(); ++i) {
            if (new_cells[i].h && new_cells[i].w)
                matrix.push_back(new_cells[i]);
        }

        for (int c = 0; c < matrix.size(); ++c) {
            matrix[c].idx = c;
        }
    }


    void insert_images(vector<int>& data, vector<Fit>& fits, vector<Cell>& fitting_matrix, vector<Img>& images) {
        vector<Fit> used;

        int currently_added = 0;
        int max_added = 200;

        for (int i = 0; i < fits.size(); ++i) {
            Fit& fit = fits[i];
            Img& img_data = images[fit.img_idx];
            int h = fit.h;
            int w = fit.w;

            bool intersecting = false;
            for (int u = 0; u < used.size(); ++u) {
                Img& img_used = images[used[u].img_idx];
                if (plane_intersect(used[u].row, img_used.h, used[u].column, img_used.w,
                                    fit.row, h, fit.column, w)) {
                    intersecting = true;
                    break;
                }
                if (img_used.idx == fit.img_idx) {
                    intersecting = true;
                    break;
                }
            }

            if (intersecting)
                continue;

            ++currently_added;
            if (currently_added > max_added)
                break;

            DBUG2("First selected image: ", img_data.idx)
            DBUG2("   score: ", fit.score)

            Cell cell;
            cell.idx = 0;
            cell.used = 1;
            cell.skeep = 1;
            cell.top_row = fit.row;
            cell.left_col = fit.column;
            cell.bottom_row = cell.top_row + h - 1;
            cell.right_col = cell.left_col + w - 1;
            cell.h = h;
            cell.w = w;
            cell.img_offset = img_data.offset;
            cell.img_num = img_data.idx;
            cell.score = fit.score;
            add_cell(fitting_matrix, cell);

            img_data.cells[cell.idx] = cell.score;
            img_data.used_in = cell.idx;
            img_data.skeep = 1;

            used.push_back(fit);
        }
    }

    void replace_fit(vector<Fit>& fits, Fit& fit) {
        int min_idx = 0;
        double min_score = fits[min_idx].score;

        for (int i = 1; i < fits.size(); ++i) {
            if (fits[i].score < min_score) {
                min_idx = i;
                min_score = fits[min_idx].score;
            }
        }

        if (min_score < fit.score)
            fits[min_idx] = fit;
    }

    void smart_fit_target_1(vector<int>& data, int& cur_idx, vector<int>& result) {
        vector<Img> images;
        images.reserve(1024 * 10);
        prepare_images(data, cur_idx, images);

        vector<Cell> fitting_matrix;
        fitting_matrix.reserve(1024 * 16);

        DBUG2("GRID_H: ", GRID_H)
        DBUG2("GRID_W: ", GRID_W)

        prepare_matrix(fitting_matrix);
        DBUG2("Matrix: ", fitting_matrix.size())

        vector<Fit> fits;
        fits.reserve(1024 * 512);

        for (int i = 0; i < images.size(); ++i) {
            Img& img_data = images[i];

            vector<Fit> best_fits;
            int max_fits = 30;

            double score = 0.;
            int row = 0;
            int column = 0;
            get_best_position(data, img_data.offset, 5, row, column, score, best_fits, max_fits, i);
            int h = img_data.h;
            int w = img_data.w;

            DBUG2("Img #", i)

            double k = 1.;

            for (int n = 0; n < 5; ++n) {
                k -= .1;
                int new_h = img_data.h * k;
                int new_w = img_data.w * k;

                scale(data, img_data.offset, img_data.h, img_data.w, new_h, new_w, new_img_1);
                double score2 = 0.;
                int row2 = 0;
                int column2 = 0;
                get_best_position2(new_img_1, new_h, new_w, 15, row2, column2, score2, best_fits, max_fits, i);
            }

            for (int n = 0; n < best_fits.size(); ++n)
                fits.push_back(best_fits[n]);
        }
        std::sort(fits.begin(), fits.end(), cmp_fit_desc());

        insert_images(data, fits, fitting_matrix, images);

        #if defined DEBUG
            for (int c = 0; c < fitting_matrix.size(); ++c) {
                DBUG2("Cell #", c)
                DBUG2("  idx: ", fitting_matrix[c].idx)
                DBUG2("  row: ", fitting_matrix[c].top_row)
                DBUG2("  col: ", fitting_matrix[c].left_col)
                DBUG2("  h: ", fitting_matrix[c].h)
                DBUG2("  w: ", fitting_matrix[c].w)
            }
        #endif


        int num_of_imgs = 200;
        vector<int> indices;
        indices.reserve(num_of_imgs);
        get_random_indices(indices, num_of_imgs, 200);

        // go through the imeges and find the best fit
        for (int i = 0; i < indices.size(); ++i) {
            Img& img_data = images[indices[i]];
            int offset = img_data.offset + 2;

            if (0 == img_data.skeep) {
                //fill_scores_approx(data, img_data.offset, fitting_matrix, img_data);
                fill_scores_scale(data, img_data.offset, fitting_matrix, img_data);
            }
        }

        // merge
        vector<int> selected;
        selected.reserve(200);

        Beam<Img, Cell> beam;
        beam.best_layout(images, fitting_matrix, selected);

        result.assign(800, -1);
        for (int c = 0; c < fitting_matrix.size(); ++c) {
            Img& img_data = images[selected[c]];

            img_data.used_in = c;
            int idx = img_data.idx * 4;
            result[idx + 0] = fitting_matrix[c].top_row;
            result[idx + 1] = fitting_matrix[c].left_col;
            result[idx + 2] = fitting_matrix[c].bottom_row;
            result[idx + 3] = fitting_matrix[c].right_col;
        }
    }


public:

    vector<int> compose(vector<int>& data) {
        int next_img = get_target(data);

        GRID_H = 58;
        GRID_W = 58;

        vector<int> result;
        result.reserve(800 * 8);
        smart_fit_target_1(data, next_img, result);

        return result;
    }


};



#if defined DEBUG

void read_from_cin(vector<int>& data, int& len);

int main(int argc, const char* argv[]) {
    vector<int> data;
    int len;

    read_from_cin(data, len);

    // some debug info
    DBUG2("Read len: ", len)
    DBUG1("")
    DBUG1("Data")
    //DBUG_ARR(data)
    DBUG1("")

    // process
    CollageMaker cmpsr;

    clock_t start = clock();
    vector<int> result = cmpsr.compose(data);
    clock_t d = clock() - start;
    DBUG2("Time (sec): ", ((float)d)/CLOCKS_PER_SEC)

    // output the result
    for (int i = 0; i < 800; ++i) {
        cout << result[i] << endl;
    }
    cout.flush();

    DBUG1("---DONE---")

    return 0;
}

void read_from_cin(vector<int>& data, int& len) {
    cin >> len;
    data.reserve(len);
    for (int i = 0; i < len; ++i) {
        int p;
        cin >> p;
        data.push_back(p);
    }
}



#endif  // if DEBUG




