#!/usr/bin/env python3
# Will McGrath
import os
import sys


class BAUsingDocker(object):
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.realpath(__file__))

    def main(self):
        pass


if __name__ == "__main__":
    sys.exit(BAUsingDocker().main())
