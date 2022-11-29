import pytest
from py.log import Producer
from xdist.scheduler import LoadScheduling

from itertools import cycle
import re


class TwisterAlphabeticalScheduling(LoadScheduling):
    """
    Try to execute tests in alphabetical order.
    """
    def __init__(self, config, log=None):
        super(TwisterAlphabeticalScheduling, self).__init__(config, log)
        if log is None:
            self.log = Producer("fairched")
        else:
            self.log = log.loadfilesched

    def check_schedule(self, node, duration=0):
        if node.shutting_down:
            return

        if self.pending:
            items_per_node_min = 2
            node_pending = self.node2pending[node]
            if len(node_pending) < items_per_node_min:
                self._send_tests(node, 1)
        else:
            node.shutdown()

        self.log("num items waiting for node:", len(self.pending))

    def schedule(self):
        # Initial distribution already happened, reschedule on all nodes
        if self.collection is not None:
            for node in self.nodes:
                self.check_schedule(node)
            return

        # XXX allow nodes to have different collections
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        # Collections are identical, create the index of pending items.
        self.collection = list(self.node2collection.values())[0]
        collection_alphabetical = self.collection.copy()
        collection_alphabetical.sort()
        for test_name in collection_alphabetical:
            self.pending.append(self.collection.index(test_name))
        if not self.collection:
            return

        initial_batch = 2 * len(self.nodes)

        nodes = cycle(self.nodes)
        for i in range(initial_batch):
            self._send_tests(next(nodes), 1)

        if not self.pending:
            # initial distribution sent all tests, start node shutdown
            for node in self.nodes:
                node.shutdown()


class TwisterOneWorkerPerPlatformScheduling(LoadScheduling):
    """
    FIXME: IT DOES NOT WORK YET

    Scheduler which tries to execute tests with schema: one platform per one
    node/worker/process.
    """
    def __init__(self, config, log=None):
        super(TwisterOneWorkerPerPlatformScheduling, self).__init__(config, log)
        if log is None:
            self.log = Producer("fairched")
        else:
            self.log = log.loadfilesched

        self.platform_tests = {}

    def check_schedule(self, node, duration=0):
        """Maybe schedule new items on the node

        If there are any globally pending nodes left then this will
        check if the given node should be given any more tests.  The
        ``duration`` of the last test is optionally used as a
        heuristic to influence how many tests the node is assigned.
        """
        if node.shutting_down:
            return

        if self.pending:
            items_per_node_min = 2
            node_pending = self.node2pending[node]
            if len(node_pending) < items_per_node_min:
                self._send_tests(node, 1)
        else:
            node.shutdown()

        self.log("num items waiting for node:", len(self.pending))

    def schedule(self):
        # Initial distribution already happened, reschedule on all nodes
        if self.collection is not None:
            for node in self.nodes:
                self.check_schedule(node)
            return

        # XXX allow nodes to have different collections
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        # Collections are identical, create the index of pending items.
        self.collection = list(self.node2collection.values())[0]
        self.pending[:] = range(len(self.collection))

        for test_name in self.collection:
            platform_name = re.search(r"platform_.+", test_name).group(0)
            if platform_name in self.platform_tests:
                self.platform_tests[platform_name].append(test_name)
            else:
                self.platform_tests[platform_name] = [test_name]

        if not self.collection:
            return

        # nodes = cycle(self.nodes)
        # for test_names in self.platform_tests.values():
        #
        # for node in self.nodes:
        #     self._send_tests(next(nodes), 1)

        if not self.pending:
            # initial distribution sent all tests, start node shutdown
            for node in self.nodes:
                node.shutdown()

    def _send_tests(self, node, num):
        tests_per_node = self.pending[:num]
        if tests_per_node:
            del self.pending[:num]
            self.node2pending[node].extend(tests_per_node)
            node.send_runtest_some(tests_per_node)


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_make_scheduler(config, log):
    return TwisterAlphabeticalScheduling(config, log)
    # return TwisterOneWorkerPerPlatformScheduling(config, log)
