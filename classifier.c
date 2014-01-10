#define _FILE_OFFSET_BITS 64



#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <heap.h>



//
// gcc -g -std=c99 -I. heap.o classifier.c -o classifier.exe
// gcc -g -std=c99 -DAS_LIB -I. heap.o classifier.c -shared -o classifier.dll
//

#define PYT_L_ALLIGN 8
#define NUM_ITEMS 65000000
#define BUFFER_SIZE (NUM_ITEMS*PYT_L_ALLIGN)

#define NUMBER_K 5


#define BAD_SIZE -1
#define BAD_NUM_ITEMS -2
#define BAD_NUM_WORDS -3
#define BAD_NUM_KW -4
#define BAD_PACK_SIZE -5


#define START 0
#define ID 1
#define T_NUM_WORDS 2
#define T_WORD 3
#define T_NUMBER 4
#define NUM_WORDS 5
#define WORD 6
#define NUMBER 7
#define NUM_KW 8
#define KWORDS 9




double FULLY_CLASSIFIED_THRESHOLD = 0.8;



struct STAT {
    long number_of_packs;
};

struct MSG {
    long id;

    long t_num_words;
    long t_cur_word;
    double *t_words;

    long num_words;
    long cur_word;
    double *words;

    // flag to avoid comparing two instances of the same message in case of cross testing
    int cross_testing;

    // flag shows whether the message has already been fully classified
    int classified;

    //
    int t_calc;
    int calc;

    double t_sum;
    double sum;
    double similarity;

    struct heap* topK;
};

struct PACK {
    long state;
    long number_items;
    long id;

    long t_num_words;
    long t_cur_word;

    long num_words;
    long cur_word;

    long num_kw;

    long key_words_size;
    double* key_words;
};

//
// declarations
//

struct PACK* allocate_pack();
void free_pack(struct PACK* p);
int process(struct MSG* messages, int messages_size, struct PACK* pack, unsigned char* buffer, size_t size, struct STAT* stat);


//
// LIB
//

__declspec(dllexport)
void set_fully_classified_threshold(double val) {
    FULLY_CLASSIFIED_THRESHOLD = val;
}

__declspec(dllexport)
struct MSG* allocate_messages(int size) {
    struct MSG* p = (struct MSG*)malloc(size * sizeof(struct MSG));
    memset(p, 0, size * sizeof(struct MSG));

    for (int i = 0; i < size; ++i) {
        struct MSG* msg = &p[i];
        msg->topK = allocate_heap(NUMBER_K);
    }

    return p;
}

__declspec(dllexport)
void free_messages(struct MSG* p, int size) {
    for (int i = 0; i < size; ++i) {
        struct MSG* msg = &p[i];
        free_heap(msg->topK);
        //if (msg->t_words)
        //   free(msg->t_words);
        //if (msg->words)
        //    free(msg->words);
    }
    free(p);
}

__declspec(dllexport)
void add_message(void* messages, long idx, long id, long t_num_words, double* t_words, long num_words, double* words, int cross_testing) {
    struct MSG* msg = (struct MSG*)messages + idx;

    msg->id = id;

    msg->t_num_words = t_num_words;
    msg->t_cur_word = 0;
    msg->t_words = t_words;
    //msg->t_words = (double*)malloc(t_num_words * 2 * sizeof(double));
    //memcpy(msg->t_words, t_words, t_num_words * 2 * sizeof(double));

    msg->num_words = num_words;
    msg->cur_word = 0;
    msg->words = words;
    //msg->words =(double*)malloc(num_words * 2 * sizeof(double));
    //memcpy(msg->words, words, num_words * 2 * sizeof(double));

    msg->cross_testing = cross_testing;
    msg->classified = 0;

    msg->t_calc = 0;
    msg->calc = 0;

    msg->t_sum = 0.0;
    msg->sum = 0.0;
}

__declspec(dllexport)
void get_message_key_words(struct MSG* messages, int idx, int** key_words, int* size) {
    struct MSG* msg = &messages[idx];

    *size = heap_size(msg->topK) * 2;
    *key_words = (int*)malloc(*size * sizeof(int));

    for (int i = 0; i < *size/2; ++i) {
        struct heap_node* n = &msg->topK->array[i];
        (*key_words)[i*2+0] = (int)n->data;
        (*key_words)[i*2+1] = (int)(n->key*100);
    }
}

__declspec(dllexport)
void free_message_key_words(int* key_words) {
    free(key_words);
}

__declspec(dllexport)
int get_message_id(struct MSG* messages, int idx) {
    struct MSG* msg = &messages[idx];
    return msg->id;
}


__declspec(dllexport)
void classify(struct MSG* messages, int messages_size, const char* model, int* error, int* num_of_packs) {
    FILE *fd = fopen(model, "rb");

    unsigned char* buffer = (unsigned char*)malloc(BUFFER_SIZE);

    struct STAT stat;
    memset(&stat, 0, sizeof(struct STAT));

    struct PACK* pack = allocate_pack();

    while (1) {
        size_t items_read = fread(buffer, PYT_L_ALLIGN, NUM_ITEMS, fd);
#if defined DEBUG
        printf("Read items: %d\n", items_read);
#endif
        if (items_read < NUM_ITEMS) {
#if defined DEBUG
            printf("WARN EOF: %d, ERR: %d\n", feof(fd), ferror(fd));
#endif
        }
        if (!items_read)
            break;
        *error = process(messages, messages_size, pack, buffer, items_read*PYT_L_ALLIGN, &stat);
        switch (*error) {
        case BAD_SIZE:
#if defined DEBUG
            printf("#ERROR: bad size %d\n", items_read*PYT_L_ALLIGN);
#endif
            break;
        case BAD_NUM_ITEMS:
#if defined DEBUG
            printf("#ERROR: bad num items %d\n", pack->number_items);
#endif
            break;
        case BAD_NUM_WORDS:
#if defined DEBUG
            printf("#ERROR: bad num words %d\n", pack->num_words);
#endif
            break;
        case BAD_NUM_KW:
#if defined DEBUG
            printf("#ERROR: bad num kw %d\n", pack->num_kw);
#endif
            break;
        case BAD_PACK_SIZE:
#if defined DEBUG
            printf("#ERROR: bad pack size %d\n", pack->number_items);
#endif
            break;

        default:
            break;
        }
    }

    free_pack(pack);
    free(buffer);
    fclose(fd);

    *num_of_packs = stat.number_of_packs;
}




//
// utils
//

struct PACK* allocate_pack() {
    struct PACK* pack = (struct PACK*)malloc(sizeof(struct PACK));
    memset(pack, 0, sizeof(struct PACK));
}

void free_pack(struct PACK* p) {
    free(p);
}


int process(struct MSG* messages, int messages_size, struct PACK* pack, unsigned char* buffer, size_t size, struct STAT* stat) {
    if (!size || 0 != size % PYT_L_ALLIGN)
        return BAD_SIZE;

    size_t offset = 0;

    while (offset < size) {
        switch (pack->state) {
        case START:
            pack->number_items = (long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            if (!pack->number_items)
                return BAD_NUM_ITEMS;
            pack->state = ID;

#if defined DEBUG
            printf(" number of items: %d\n", pack->number_items);
#endif
            break;

        case ID:
            pack->id = (long)(long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            --pack->number_items;
            pack->state = T_NUM_WORDS;

#if defined DEBUG
            printf(" id: %d\n", pack->id);
#endif
            break;

        // TITLE
        case T_NUM_WORDS:
            pack->t_num_words = (long)(long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            if (pack->t_num_words < 0)
                return BAD_NUM_WORDS;
            --pack->number_items;
            pack->t_cur_word = 0;
            if (pack->t_num_words)
                pack->state = T_WORD;
            else
                pack->state = NUM_WORDS;

#if defined DEBUG
            printf(" number of t words: %d\n", pack->t_num_words);
#endif
            break;

        case T_WORD:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];

                    // skip if already classified
                    if (msg->classified)
                        continue;

                    if (msg->t_cur_word < msg->t_num_words) {
                        while (msg->t_cur_word < msg->t_num_words && msg->t_words[msg->t_cur_word*2 + 0] < *pd) {
                            msg->t_cur_word += 1;
                        }

                        if (msg->t_cur_word < msg->t_num_words && msg->t_words[msg->t_cur_word*2 + 0] == *pd) {
                            msg->t_calc = 1;
                        }else{
                            msg->t_calc = 0;
                        }
                    }
                } // for messages

                pack->state = T_NUMBER;

#if defined DEBUG
                printf(" t word: %5.2f\n", *pd);
#endif
            }
            break;

        case T_NUMBER:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];
                    if (msg->t_calc) {
                        msg->t_calc = 0;
                        double tmp = *pd * msg->t_words[msg->t_cur_word*2 + 1];
                        msg->t_sum += tmp;
#if defined DEBUG
                        printf(" (t word match [%d] %5.2f %5.2f)\n", i, msg->t_words[msg->t_cur_word*2 + 0], msg->t_words[msg->t_cur_word*2 + 1]);
#endif
                    }
                } // for messages

                pack->t_cur_word += 1;
                if (pack->t_cur_word < pack->t_num_words)
                    pack->state = T_WORD;
                else
                    pack->state = NUM_WORDS;

#if defined DEBUG
                printf(" t number: %5.2f\n", *pd);
#endif
            }
            break;

        // BODY
        case NUM_WORDS:
            pack->num_words = (long)(long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            if (pack->num_words < 0)
                return BAD_NUM_WORDS;
            --pack->number_items;
            pack->cur_word = 0;
            if (pack->num_words)
                pack->state = WORD;
            else
                pack->state = NUM_KW;

#if defined DEBUG
            printf(" number of words: %d\n", pack->num_words);
#endif
            break;

        case WORD:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];

                    // skip if already classified
                    if (msg->classified)
                        continue;

                    if (msg->cur_word < msg->num_words) {
                        while (msg->cur_word < msg->num_words && msg->words[msg->cur_word*2 + 0] < *pd) {
                            msg->cur_word += 1;
                        }

                        if (msg->cur_word < msg->num_words && msg->words[msg->cur_word*2+0] == *pd) {
                            msg->calc = 1;
                        }else{
                            msg->calc = 0;
                        }
                    }
                } // for messages

                pack->state = NUMBER;

#if defined DEBUG
                printf(" word: %5.2f\n", *pd);
#endif
            }
            break;

        case NUMBER:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];
                    if (msg->calc) {
                        msg->calc = 0;
                        double tmp = *pd * msg->words[msg->cur_word*2 + 1];
                        msg->sum += tmp;
#if defined DEBUG
                        printf(" (word match [%d] %5.2f %5.2f)\n", i, msg->words[msg->cur_word*2 + 0], msg->words[msg->cur_word*2 + 1]);
#endif
                    }
                } // for messages

                pack->cur_word += 1;
                if (pack->cur_word < pack->num_words)
                    pack->state = WORD;
                else
                    pack->state = NUM_KW;

#if defined DEBUG
                printf(" number: %5.2f\n", *pd);
#endif
            }
            break;

        case NUM_KW:
            pack->num_kw = (long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            --pack->number_items;

            pack->state = KWORDS;

            // finish calc _before_ I start processing tags
            for (int i = 0; i < messages_size; ++i) {
                struct MSG* msg = &messages[i];

                // skip if already classified
                if (msg->classified)
                    continue;

                double total = msg->t_sum * .60 + msg->sum * .40;
#if defined DEBUG
                printf(" [%d] Title similarity %d vs %d: %5.2f\n", i, pack->id, msg->id, msg->t_sum);
                printf(" [%d] Body similarity %d vs %d: %5.2f Total: %5.2f\n", i, pack->id, msg->id, msg->sum, total);
#endif
                msg->similarity = total;
            }



#if defined DEBUG
            printf(" number of kw: %d\n", pack->num_kw);
#endif
            break;

        case KWORDS:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                --pack->num_kw;

                // preserve the tag if needed
                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];

                    // skip if already classified
                    if (msg->classified) {
                        if (msg->classified == pack->id) {
                            // it's classified from this pack
                            if (!msg->cross_testing || msg->cross_testing && msg->id != pack->id)
                                heap_add(msg->topK, msg->similarity, (void*)(int)*pd);
                        }else{
                            continue;
                        }
                    }else{
                        if (!msg->cross_testing || msg->cross_testing && msg->id != pack->id) {

                            if (msg->similarity >= FULLY_CLASSIFIED_THRESHOLD) {
                                msg->classified = pack->id;
                                // remove everything previously accumulated
                                while (heap_size(msg->topK))
                                    heap_pop(msg->topK);
                            }


                            heap_add(msg->topK, msg->similarity, (void*)(int)*pd);
                        }
                    }
                }

#if defined DEBUG
                printf(" kw: %5.2f\n", *pd);
#endif

                if (!pack->num_kw) {
                    // sanity check
                    if (pack->number_items) {
                        return BAD_PACK_SIZE;
                    }

                    // reset pack
                    memset(pack, 0, sizeof(struct PACK));

                    // reset msg too
                    for (int i = 0; i < messages_size; ++i) {
                        struct MSG* msg = &messages[i];
                        msg->t_cur_word = 0;
                        msg->cur_word = 0;
                        msg->t_calc = 0;
                        msg->calc = 0;
                        msg->t_sum = 0;
                        msg->sum = 0;
                        msg->similarity = 0;
                    }

                    stat->number_of_packs += 1;
#if defined DEBUG
                    printf("--end of pack--\n");
#endif
                }
            }
            break;
        } // switch
    } // while

    return 0;
}



#if !defined AS_LIB

int main() {
    //1.0, 0.65079137345596849, 21.0, 0.75925660236529657, 3.0, 1.0, 0.55011156171227904, 3.0, 0.27505578085613952, 21.0, 0.78849323845426655

    double t_words[] = {1.0, 0.65079137345596849, 21.0, 0.75925660236529657};
    double words[] = {1.0, 0.55011156171227904, 3.0, 0.27505578085613952, 21.0, 0.78849323845426655};

    double t_words2[] = {2.0, 0.65079137345596849, 12.0, 0.75925660236529657};
    double words2[] = {3.0, 0.55011156171227904, 56.0, 0.27505578085613952, 77.0, 0.78849323845426655};

    //
    int messages_size = 2;
    struct MSG* messages = allocate_messages(messages_size);

    add_message(messages, 0, 1,
                sizeof(t_words) / sizeof(double) / 2, t_words,
                sizeof(words) / sizeof(double) / 2, words,
                0);

    add_message(messages, 1, 2,
                sizeof(t_words2) / sizeof(double) / 2, t_words2,
                sizeof(words2) / sizeof(double) / 2, words2,
                0);

    int number_of_packs;
    int error;
    classify(messages, messages_size, "..\\processed\\Model.bin", &error, &number_of_packs);

    switch (error) {
    case BAD_SIZE:
        printf("#ERROR: bad size\n");
        break;
    case BAD_NUM_ITEMS:
        printf("#ERROR: bad num items\n");
        break;
    case BAD_NUM_WORDS:
        printf("#ERROR: bad num words\n");
        break;
    case BAD_NUM_KW:
        printf("#ERROR: bad num kw\n");
        break;
    case BAD_PACK_SIZE:
        printf("#ERROR: bad pack size\n");
        break;

    default:
        break;
    }



    int *tags;
    int size;

    get_message_key_words(messages, 0, &tags, &size);
    printf("Keywords[0]:\n");
    for (int i = 0; i < size; ++i) {
        printf(" %d\n", tags[i]);
    }
    free_message_key_words(tags);

    get_message_key_words(messages, 1, &tags, &size);
    printf("Keywords[1]:\n");
    for (int i = 0; i < size; ++i) {
        printf(" %d\n", tags[i]);
    }
    free_message_key_words(tags);


    // print topKs
    for (int i = 0; i < messages_size; ++i) {
        struct MSG* msg = &messages[i];
        printf("Top K [%d]\n", i);
        while (heap_size(msg->topK)) {
            struct heap_node* n = heap_top(msg->topK);
            printf(" %d %5.2f\n", (long)n->data, n->key);
            heap_pop(msg->topK);
        }

    }


    free_messages(messages, messages_size);

    printf("Finished: number of packs checked %d\n", number_of_packs);

    return 0;
}

#endif // AS_LIB
