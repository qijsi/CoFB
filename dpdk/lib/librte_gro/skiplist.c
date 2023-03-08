#include "skiplist.h"

SkiplistNode* skiplistNodeCreate(int level, uint32_t key, uint32_t value) {
    SkiplistNode* p = (SkiplistNode *)malloc(sizeof(*p) + sizeof(struct SkiplistLevel) * level);
    p->key = key;
    p->value = value;
    return p;
}

Skiplist* skiplistCreate(void) {
    Skiplist *sl = (Skiplist *)malloc(sizeof(Skiplist));
    sl->length = 0;
    sl->level = 1;
    sl->head = skiplistNodeCreate(SKIPLIST_MAXLEVEL, INT_MIN, INT_MIN);
    for (int i=0; i < SKIPLIST_MAXLEVEL; ++i) {
        sl->head->level[i].next = NULL;
    }
    return sl;
}

bool skiplistSearch(Skiplist* obj, uint32_t key) {
    SkiplistNode *p = obj->head;
    int levelIdx = obj->level - 1;
    for(int i=levelIdx; i>=0; --i) {
        while(p->level[i].next && p->level[i].next->key < key) {
            p = p->level[i].next;
        }
        if (p->level[i].next == NULL || p->level[i].next->key > key) {
            continue;
        }
        return true;
    }
    return false;
}

bool skiplistSearchValue(Skiplist* obj, uint32_t key, uint32_t value) {
    SkiplistNode *p = obj->head;
    int levelIdx = obj->level - 1;
    for(int i=levelIdx; i>=0; --i) {
        while(p->level[i].next && p->level[i].next->key < key) {
            p = p->level[i].next;
        }

        if (p->level[i].next == NULL || p->level[i].next->key > key) {
            continue;
        }
        
        while (p->level[i].next && p->level[i].next->key == key) {
            if (p->level[i].next->value == value) return true;
            p = p->level[i].next;
        }
    }
    return false;
}

uint32_t GetSkipNodeRandomLevel(void) {
    uint32_t level = 1;
    while (rand() & 0x1) {
        ++level;
    }
    return min(level, SKIPLIST_MAXLEVEL);  
}

void skiplistAdd(Skiplist* obj, uint32_t key, uint32_t value) {
    SkiplistNode *p = obj->head;
    int levelIdx = obj->level - 1;
    struct SkiplistNode *preNodes[SKIPLIST_MAXLEVEL];
    for (int i = obj->level; i < SKIPLIST_MAXLEVEL; ++i) {
        preNodes[i] = obj->head;
    }

    for (int i = levelIdx; i >= 0; --i) {
        while(p->level[i].next && p->level[i].next->key < key) {
            p = p->level[i].next;
        }
        preNodes[i] = p;
    }

    uint32_t newLevel = GetSkipNodeRandomLevel();
    struct SkiplistNode *newNode = skiplistNodeCreate(newLevel, key, value);
    for (uint32_t i=0; i<newLevel; ++i) {
        newNode->level[i].next = preNodes[i]->level[i].next;
        preNodes[i]->level[i].next = newNode;
    }
    obj->level = max(obj->level, newLevel);
    ++obj->length;
}

void skiplistNodeDelete(Skiplist *obj, SkiplistNode *cur, SkiplistNode **preNodes) {
  uint32_t i;
  for (i = 0; i < obj->level; ++i) {
    if (preNodes[i]->level[i].next == cur) {
      preNodes[i]->level[i].next = cur->level[i].next;
    }
  }

    for (uint32_t i=obj->level-1; i>=1; --i) {
        if (obj->head->level[i].next != NULL) {
            break;
        }
        --obj->level;
    }
    --obj->length;
    free(cur);
}

bool skiplistErase(Skiplist* obj, uint32_t key) {
    SkiplistNode *p = obj->head;
    int levelIdx = obj->level - 1;
    struct SkiplistNode *preNodes[SKIPLIST_MAXLEVEL];
    for (int i=levelIdx; i>=0; --i) {
        while(p->level[i].next && p->level[i].next->key < key) {
            p = p->level[i].next;
        }
        preNodes[i] = p;
    }

    p = p->level[0].next;
    if (p && p->key == key) {
        skiplistNodeDelete(obj, p, preNodes);
        return true;
    }
    return false;
}

bool skiplistEraseFirst(Skiplist* obj) {
    uint32_t key = obj->head->level[0].next->key;
    return skiplistErase(obj, key);
}

bool skiplistEraseNth(Skiplist* obj, uint32_t n) {
  SkiplistNode *p = obj->head;
  uint32_t i = 0;
  while (i<n && NULL != p) {
    p = p->level[0].next;
    i++;
  }

 // printf("delete key:%u, value:%u\n", p->key, p->value);
  return skiplistErase(obj, p->key);
}


void skiplistTraverse(Skiplist* obj) {
    SkiplistNode *p = obj->head->level[0].next;
    while(NULL != p) {
      printf("%u ", p->value);
      p = p->level[0].next;
    }
    printf("\n");
}

uint32_t skiplistFirst(Skiplist* obj) {
  return obj->head->level[0].next->value;
}

uint32_t skiplistNth(Skiplist* obj, uint32_t n) {
  uint32_t i = 0;
  SkiplistNode *p = obj->head;

  if (obj->length < n) return INVALID_VALUE;

  while(i < n && NULL != p->level[0].next) {
    p = p->level[0].next;
    i++;
  }

  return p->value;
}

void skiplistFree(Skiplist* obj) {
    SkiplistNode *cur = obj->head->level[0].next;
    SkiplistNode *d;
    while (cur) {
        d = cur;
        cur = cur->level[0].next;
        free(d);
    }
    free(obj->head);
    free(obj);
}

#if 0
void print(Skiplist *obj) {
	for(int i=0; i<obj->level - 1; i++) {
    SkiplistNode *p = obj->head->level[i].next;
    while(p!=NULL) {
        printf("%d ", p->value);
		p = p->level[i].next;
    }
    printf("\n");
	}
	printf("\n");
}
#endif

#if 0
int main() {
    Skiplist *list = skiplistCreate();
    skiplistAdd(list, 5);
    skiplistAdd(list, 10);
    skiplistAdd(list, 15);
    skiplistAdd(list, 8);
    skiplistAdd(list, 2);
    skiplistAdd(list, 12);
    print(list);
    skiplistErase(list, 8);
    print(list);
    skiplistErase(list, 12);
    print(list);
    skiplistFree(list);
}
#endif
