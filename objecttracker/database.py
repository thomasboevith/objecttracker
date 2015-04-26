# coding: UTF-8
import os
import tempfile
import sqlite3

import logging
# Define the logger
LOG = logging.getLogger(__name__)

DB_FILE = os.path.join(tempfile.gettempdir(), "objectcounter.db")

class Db:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
        self.c = self.conn.cursor()

    def __enter__(self):
        LOG.debug("Entering db.")
        return self
    
    def __exit__(self, type, value, traceback):
        LOG.debug("Exiting db.")
        self.conn.close()
        
    def execute(self, sql, values=None):
        if values == None:
            LOG.debug("Executing SQL: '%s'."%(sql))
            self.c.execute(sql)
        else: 
            LOG.debug("Executing SQL: '%s' with values: '%s'."%(sql, values))
            self.c.execute(sql, values)

        LOG.debug("Committing SQL.")
        self.conn.commit()
        
