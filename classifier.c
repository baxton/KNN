

#define _FILE_OFFSET_BITS 64



#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <heap.h>



//
// gcc -g -std=c99 -I. heap.o classifier.c -o classifier.exe
//

#define PYT_L_ALLIGN 8
#define NUM_ITEMS 512
#define BUFFER_SIZE (NUM_ITEMS*PYT_L_ALLIGN)

#define NUMBER_K 3


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


struct MSG {
    long id;

    long t_num_words;
    long t_cur_word;
    double *t_words;

    long num_words;
    long cur_word;
    double *words;

    //
    int t_calc;
    int calc;

    double t_sum;
    double sum;

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
};




int process(struct MSG* messages, int messages_size, struct PACK* pack, unsigned char* buffer, size_t size) {
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

            printf(" number of items: %d\n", pack->number_items);
            break;

        case ID:
            pack->id = (long)(long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            --pack->number_items;
            pack->state = T_NUM_WORDS;

            printf(" id: %d\n", pack->id);
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

            printf(" number of t words: %d\n", pack->t_num_words);
            break;

        case T_WORD:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];
                    if (msg->t_cur_word < msg->t_num_words) {
                        while (msg->t_cur_word < msg->t_num_words && msg->t_words[msg->t_cur_word*2 + 0] < *pd) {
                            ++msg->t_cur_word;
                        }

                        if (msg->t_cur_word < msg->t_num_words && msg->t_words[msg->t_cur_word*2+0] == *pd) {
                            msg->t_calc = 1;
                        }else{
                            msg->t_calc = 0;
                        }
                    }
                } // for messages

                pack->state = T_NUMBER;

                printf(" t word: %5.2f\n", *pd);
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
                        printf(" (t word match [%d] %5.2f %5.2f)\n", i, msg->t_words[msg->t_cur_word*2 + 0], msg->t_words[msg->t_cur_word*2 + 1]);
                    }
                } // for messages

                ++pack->t_cur_word;
                if (pack->t_cur_word < pack->t_num_words)
                    pack->state = T_WORD;
                else
                    pack->state = NUM_WORDS;

                printf(" t number: %5.2f\n", *pd);
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

            printf(" number of words: %d\n", pack->num_words);
            break;

        case WORD:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                for (int i = 0; i < messages_size; ++i) {
                    struct MSG* msg = &messages[i];
                    if (msg->cur_word < msg->num_words) {
                        while (msg->cur_word < msg->num_words && msg->words[msg->cur_word*2 + 0] < *pd) {
                            ++msg->cur_word;
                        }

                        if (msg->cur_word < msg->num_words && msg->words[msg->cur_word*2+0] == *pd) {
                            msg->calc = 1;
                        }else{
                            msg->calc = 0;
                        }
                    }
                } // for messages

                pack->state = NUMBER;

                printf(" word: %5.2f\n", *pd);
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
                        printf(" (word match [%d] %5.2f %5.2f)\n", i, msg->words[msg->cur_word*2 + 0], msg->words[msg->cur_word*2 + 1]);
                    }
                } // for messages

                ++pack->cur_word;
                if (pack->cur_word < pack->num_words)
                    pack->state = WORD;
                else
                    pack->state = NUM_KW;

                printf(" number: %5.2f\n", *pd);
            }
            break;

        case NUM_KW:
            pack->num_kw = (long)*(double*)(buffer + offset);
            offset += PYT_L_ALLIGN;
            --pack->number_items;

            pack->state = KWORDS;

            printf(" number of kw: %d\n", pack->num_kw);
            break;

        case KWORDS:
            {
                double *pd = (double*)(buffer + offset);
                offset += PYT_L_ALLIGN;
                --pack->number_items;

                --pack->num_kw;

                printf(" kw: %5.2f\n", *pd);

                if (!pack->num_kw) {
                    // sanity check
                    if (pack->number_items) {
                        return BAD_PACK_SIZE;
                    }

                    // finish calc
                    for (int i = 0; i < messages_size; ++i) {
                        struct MSG* msg = &messages[i];
                        double total = msg->t_sum * .60 + msg->sum * .40;
                        printf(" [%d] Title similarity %d vs %d: %5.2f\n", i, pack->id, msg->id, msg->t_sum);
                        printf(" [%d] Body similarity %d vs %d: %5.2f Total: %5.2f\n", i, pack->id, msg->id, msg->sum, total);

                        if (NUMBER_K == heap_size(msg->topK)) {
                            struct heap_node* n = heap_top(msg->topK);
                            if (n->key < total) {
                                heap_pop(msg->topK);
                                heap_add(msg->topK, total, (void*)pack->id);
                            }
                        }else
                            heap_add(msg->topK, total, (void*)pack->id);
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
                    }

                    printf("--end of pack--\n");
                }
            }
            break;
        } // switch
    } // while
}





int main() {
    FILE *fd = fopen("python_arr.txt", "rb");

    unsigned char buffer[BUFFER_SIZE];


    //1.0, 0.65079137345596849, 21.0, 0.75925660236529657, 3.0, 1.0, 0.55011156171227904, 3.0, 0.27505578085613952, 21.0, 0.78849323845426655

    double t_words[] = {1.0, 0.65079137345596849, 21.0, 0.75925660236529657};
    double words[] = {1.0, 0.55011156171227904, 3.0, 0.27505578085613952, 21.0, 0.78849323845426655};

    double t_words2[] = {2.0, 0.65079137345596849, 12.0, 0.75925660236529657};
    double words2[] = {3.0, 0.55011156171227904, 56.0, 0.27505578085613952, 77.0, 0.78849323845426655};

    struct MSG msg;
    memset(&msg, 0, sizeof(struct MSG));
    msg.t_num_words = sizeof(t_words) / sizeof(double) / 2;
    msg.t_words = t_words;
    msg.num_words = sizeof(words) / sizeof(double) / 2;
    msg.words = words;

    struct MSG msg2;
    memset(&msg2, 0, sizeof(struct MSG));
    msg2.t_num_words = sizeof(t_words2) / sizeof(double) / 2;
    msg2.t_words = t_words2;
    msg2.num_words = sizeof(words2) / sizeof(double) / 2;
    msg2.words = words2;

    msg.topK = allocate_heap(NUMBER_K);
    msg2.topK = allocate_heap(NUMBER_K);

    struct MSG messages[] = {msg, msg2};
    int messages_size = sizeof(messages) / sizeof(struct MSG);

    struct PACK pack;
    memset(&pack, 0, sizeof(struct PACK));

    while (1) {
        size_t items_read = fread(buffer, PYT_L_ALLIGN, NUM_ITEMS, fd);
        printf("Read items: %d\n", items_read);
	if (items_read < NUM_ITEMS) {
		printf("WARN EOF: %d, ERR: %d\n", feof(fd), ferror(fd));
	}
        if (!items_read)
            break;

        switch (process(messages, messages_size, &pack, buffer, items_read*PYT_L_ALLIGN)) {
        case BAD_SIZE:
            printf("#ERROR: bad size %d\n", items_read*PYT_L_ALLIGN);
            break;
        case BAD_NUM_ITEMS:
            printf("#ERROR: bad num items %d\n", pack.number_items);
            break;
        case BAD_NUM_WORDS:
            printf("#ERROR: bad num words %d\n", pack.num_words);
            break;
        case BAD_NUM_KW:
            printf("#ERROR: bad num kw %d\n", pack.num_kw);
            break;
        case BAD_PACK_SIZE:
            printf("#ERROR: bad pack size %d\n", pack.number_items);
            break;

        default:
            break;
        }
    }



    fclose(fd);

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

    free_heap(msg.topK);
    free_heap(msg2.topK);

    return 0;
}
