#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>

//#define bool int
//#define TRUE 1
//#define FALSE 0

#define SKIPLIST_MAXLEVEL 32    
#define INVALID_VALUE 0xffffffffUL

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct SkiplistNode {
    uint32_t key; //time_stamp
    uint32_t value; //idx in gro tbl
    struct SkiplistLevel {
        struct SkiplistNode *next;
    } level[];
} SkiplistNode;

typedef struct Skiplist {
    struct SkiplistNode *head;
    uint32_t length;
    uint32_t level;
} Skiplist;

SkiplistNode* skiplistNodeCreate(int level, uint32_t key, uint32_t value);
Skiplist* skiplistCreate(void);
bool skiplistSearch(Skiplist* obj, uint32_t key);
bool skiplistSearchValue(Skiplist* obj, uint32_t key, uint32_t value);
uint32_t GetSkipNodeRandomLevel(void);
void skiplistAdd(Skiplist* obj, uint32_t key, uint32_t value);
void skiplistNodeDelete(Skiplist *obj, SkiplistNode *cur, SkiplistNode **preNodes);
bool skiplistErase(Skiplist* obj, uint32_t key);
void skiplistFree(Skiplist* obj);
uint32_t skiplistFirst(Skiplist* obj);
uint32_t skiplistNth(Skiplist* obj, uint32_t n);
bool skiplistEraseFirst(Skiplist* obj);
bool skiplistEraseNth(Skiplist* obj, uint32_t n);
void skiplistTraverse(Skiplist* obj);
