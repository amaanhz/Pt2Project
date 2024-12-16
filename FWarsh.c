// I realised this is actually just Floyd-Warshall.....


/*
    BMF_b_args* a = (BMF_b_args*) args;
    const Graph* graph = a->graph; int** m_dist = a->m_dist;
    Result** results = a->results;

    pthread_mutex_t* q_lock = a->q_lock; pthread_mutex_t* i_lock = a->i_lock;
    pthread_mutex_t* v_locks = a->v_locks;
    // Synchronise iterations. Each dist depends on the last iteration's.
    while (1)
    {
        // check if we've reached V iterations
        pthread_mutex_lock(i_lock);
        if (*a->iter == graph->size)
        {
            pthread_mutex_unlock(i_lock);
            break;
        }
        pthread_mutex_unlock(i_lock);
        ///////////////

        // We haven't, lets get the next node to work on
        pthread_mutex_lock(q_lock);
        int src = *a->next_node;
        if (src >= graph->size)
        {
            // End of iteration, need to increment iter number
            // Need to make sure every node synchronises at this point though, and only one node makes changes
            if (pthread_mutex_trylock(i_lock) == 0)
            {
                pthread_mutex_lock(q_lock);
                *a->iter++;
                *a->next_node = 0;
                pthread_mutex_unlock(i_lock);
                pthread_mutex_unlock(q_lock);
            }
            pthread_barrier_wait(a->barrier);
            continue;
        }
        *a->next_node++;
        pthread_mutex_unlock(q_lock);

        Node* adj = graph->verts[src];
        while (adj != NULL)
        {
            // Lock source and target
            pthread_mutex_lock(v_locks + src);
            pthread_mutex_lock(v_locks + adj->vertex);
            int v = adj->vertex;

            // Perform the relaxation
            if (m_dist[src][v] != INT_MAX && m_dist[src][v] + adj->weight < m_dist[src][v])
            {
                m_dist[src][v] = m_dist[src][v] + adj->weight;
            }

            // Release locks
            pthread_mutex_unlock(v_locks + src);
            pthread_mutex_unlock(v_locks + adj->vertex);

            adj = adj->next; // Continue for all adjacent nodes
        }
    }
 */
