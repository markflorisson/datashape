# Utilities for the high level Blaze test suite

import unittest
import tempfile
import os, shutil, glob

# Useful superclass for disk-based tests
class MayBeUriTest():

    uri = False

    def setUp(self):
        if self.uri:
            prefix = 'barray-' + self.__class__.__name__
            self.rootdir = tempfile.mkdtemp(prefix=prefix)
            self.rooturi = 'blz://' + self.rootdir
            os.rmdir(self.rootdir)  # tests needs this cleared
        else:
            self.rootdir = None

    def tearDown(self):
        if self.uri:
            # Remove every directory starting with rootdir
            for dir_ in glob.glob(self.rootdir+'*'):
                shutil.rmtree(dir_)

class BTestCase(unittest.TestCase):
    """
    TestCase that provides some stuff missing in 2.6.
    """

    def assertIsInstance(self, obj, cls, msg=None):
        self.assertTrue(isinstance(obj, cls),
                        msg or "%s is not an instance of %s" % (obj, cls))

    def assertGreater(self, a, b, msg=None):
        self.assertTrue(a > b, msg or "%s is not greater than %s" % (a, b))
