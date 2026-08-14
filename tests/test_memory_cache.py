# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-aware prefix cache."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.memory_cache import (
    CacheStats,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _CacheEntry,
    _array_memory,
    _get_available_memory,
    estimate_kv_cache_memory,
)


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_config(self):
        config = MemoryCacheConfig()
        assert config.max_memory_mb is None
        assert config.max_memory_percent == 0.20
        assert config.max_entries == 1000
        assert config.enable_memory_tracking is True
        assert config.min_prefix_tokens == 128

    def test_custom_config(self):
        config = MemoryCacheConfig(
            max_memory_mb=2048,
            max_memory_percent=0.5,
            max_entries=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_memory_percent == 0.5
        assert config.max_entries == 100

    def test_invalid_memory_percent_zero(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=0.0)

    def test_invalid_memory_percent_negative(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=-0.1)

    def test_invalid_memory_percent_over_one(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=1.5)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            MemoryCacheConfig(max_entries=0)

    def test_invalid_min_prefix_tokens(self):
        with pytest.raises(ValueError, match="min_prefix_tokens"):
            MemoryCacheConfig(min_prefix_tokens=0)

    def test_compute_memory_limit_explicit(self):
        config = MemoryCacheConfig(max_memory_mb=1024)
        assert config.compute_memory_limit() == 1024 * 1024 * 1024

    def test_compute_memory_limit_auto(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=8 * 1024 * 1024 * 1024,  # 8GB
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            assert limit == 2 * 1024 * 1024 * 1024  # 25% of 8GB = 2GB

    def test_compute_memory_limit_fallback(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=0,  # Detection failed
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            # Fallback: 25% of 8GB = 2GB
            assert limit == 2 * 1024 * 1024 * 1024


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_queries(self):
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_memory_utilization(self):
        stats = CacheStats(
            current_memory_bytes=500 * 1024 * 1024,
            max_memory_bytes=1000 * 1024 * 1024,
        )
        assert stats.memory_utilization == 0.5

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d
        assert "memory_utilization" in d


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockDtype:
    """Mock dtype with size attribute."""

    def __init__(self, size: int):
        self.size = size


class MockShapeArray:
    """Mock array with shape and dtype (like MLX arrays) but no nbytes."""

    def __init__(self, shape: tuple, dtype_size: int):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock KV cache with keys/values attributes."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)


class MockStateCache:
    """Mock cache with state property."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self._keys = MockArray(key_bytes)
        self._values = MockArray(value_bytes)

    @property
    def state(self):
        return (self._keys, self._values)


class TestArrayMemory:
    """Tests for _array_memory helper (shape-based, no lazy eval trigger)."""

    def test_shape_dtype_estimation(self):
        """Verify shape*dtype.size computation without .nbytes access."""
        arr = MockShapeArray(shape=(2, 16, 128, 64), dtype_size=2)
        # 2 * 16 * 128 * 64 * 2 = 524288
        assert _array_memory(arr) == 2 * 16 * 128 * 64 * 2

    def test_fallback_to_nbytes(self):
        """Verify fallback to .nbytes when shape/dtype not available."""
        arr = MockArray(nbytes=4096)
        assert _array_memory(arr) == 4096

    def test_zero_for_unknown_object(self):
        """Return 0 for objects without shape/dtype/nbytes."""
        assert _array_memory(42) == 0
        assert _array_memory("string") == 0

    def test_shape_dtype_preferred_over_nbytes(self):
        """When both shape+dtype and nbytes exist, shape+dtype is used."""

        class DualArray:
            def __init__(self):
                self.shape = (10,)
                self.dtype = MockDtype(4)
                self.nbytes = 9999  # should NOT be used

        arr = DualArray()
        assert _array_memory(arr) == 40  # 10 * 4, not 9999

    def test_estimate_uses_shape_based_for_dict_state(self):
        """estimate_kv_cache_memory uses _array_memory (shape-based) for dicts."""
        keys = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        values = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        layer = {"state": (keys, values)}
        expected = 2 * (1 * 8 * 100 * 64 * 2)
        assert estimate_kv_cache_memory([layer]) == expected

    def test_estimate_handles_nested_cachelist_state(self):
        """Regression: DeepSeek-V4's CacheList.state nests three sub-states.

        The old two-way unpack raised ValueError, was swallowed, and the whole
        entry counted as 0 bytes — so the dashboard's Prefix Cache bar stayed
        at 0% and byte-based LRU eviction never fired for such models.
        """

        class NestedStateCache:
            def __init__(self):
                rot = (
                    MockShapeArray(shape=(1, 8, 128, 64), dtype_size=2),
                    MockShapeArray(shape=(1, 8, 128, 64), dtype_size=2),
                )
                pool_a = (
                    MockShapeArray(shape=(1, 3, 512), dtype_size=2),
                    MockShapeArray(shape=(1, 3, 128), dtype_size=2),
                    MockShapeArray(shape=(1, 40, 512), dtype_size=2),
                )
                pool_b = (None, None, None)  # empty PoolingCache members
                self.state = [rot, pool_a, pool_b]

        expected = (
            2 * (1 * 8 * 128 * 64 * 2)
            + (1 * 3 * 512 * 2)
            + (1 * 3 * 128 * 2)
            + (1 * 40 * 512 * 2)
        )
        assert estimate_kv_cache_memory([NestedStateCache()]) == expected


class TestEstimateKvCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_empty_cache(self):
        assert estimate_kv_cache_memory([]) == 0
        assert estimate_kv_cache_memory(None) == 0

    def test_cache_with_nbytes_attribute(self):
        layer = MockKVCache(1000, 1000)
        assert estimate_kv_cache_memory([layer]) == 2000

    def test_cache_with_state_property(self):
        layer = MockStateCache(500, 500)
        assert estimate_kv_cache_memory([layer]) == 1000

    def test_cache_with_dict_state(self):
        keys = MockArray(300)
        values = MockArray(300)
        layer = {"state": (keys, values)}
        assert estimate_kv_cache_memory([layer]) == 600

    def test_multiple_layers(self):
        layers = [MockKVCache(100, 100) for _ in range(4)]
        assert estimate_kv_cache_memory(layers) == 800


class TestCacheEntry:
    """Tests for _CacheEntry."""

    def test_create_entry(self):
        cache = [MockKVCache(100, 100)]
        entry = _CacheEntry.create([1, 2, 3], cache)
        assert entry.tokens == (1, 2, 3)
        assert entry.cache is cache
        assert entry.memory_bytes == 200


class TestMemoryAwarePrefixCache:
    """Tests for MemoryAwarePrefixCache."""

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def small_cache(self, model):
        """Cache with 1MB limit."""
        config = MemoryCacheConfig(
            max_memory_mb=1,
            max_entries=10,
            min_prefix_tokens=1,
        )
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KV cache with known size."""

        def _create(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        return _create

    def test_initialization(self, model):
        config = MemoryCacheConfig(max_memory_mb=100)
        cache = MemoryAwarePrefixCache(model, config)
        assert len(cache) == 0
        assert cache.memory_limit_mb == 100.0

    def test_store_and_fetch_exact_match(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3, 4, 5]
        kv = mock_kv_cache(1000)

        # Store
        assert small_cache.store(tokens, kv) is True
        assert len(small_cache) == 1

        # Fetch exact match
        result, remaining = small_cache.fetch(tokens)
        assert result is kv  # Same reference, no copy
        assert remaining == []

    def test_short_prefix_reuse_is_rejected(self, model, mock_kv_cache):
        cache = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(
                max_memory_mb=10,
                max_entries=10,
                min_prefix_tokens=8,
            ),
        )
        short_tokens = [1, 2, 3, 4, 5]
        kv = mock_kv_cache(1000)

        assert cache.store(short_tokens, kv) is False
        assert len(cache) == 0

        result, remaining = cache.fetch(short_tokens)
        assert result is None
        assert remaining == short_tokens
        assert cache.get_stats()["misses"] == 1

    def test_fetch_prefix_match(self, small_cache, mock_kv_cache):
        # Store shorter sequence
        short_tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(short_tokens, kv)

        # Fetch longer sequence that starts with cached prefix
        long_tokens = [1, 2, 3, 4, 5, 6]
        result, remaining = small_cache.fetch(long_tokens)

        assert result is kv
        assert remaining == [4, 5, 6]

    def test_fetch_miss(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)

        # Fetch completely different sequence
        result, remaining = small_cache.fetch([7, 8, 9])
        assert result is None
        assert remaining == [7, 8, 9]

    def test_lru_eviction_on_memory_pressure(self, model, mock_kv_cache):
        # Create cache with 500KB limit
        config = MemoryCacheConfig(
            max_memory_mb=0.5,
            max_entries=100,
            min_prefix_tokens=1,
        )
        cache = MemoryAwarePrefixCache(model, config)

        # Store entries that together exceed limit
        # Each is ~200KB
        for i in range(5):
            tokens = list(range(i * 10, (i + 1) * 10))
            kv = mock_kv_cache(200 * 1024)
            cache.store(tokens, kv)

        # Should have evicted older entries
        assert cache.memory_usage_mb <= 0.5
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_lru_order_updated_on_fetch(self, small_cache, mock_kv_cache):
        # Store two entries
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv1 = mock_kv_cache(100 * 1024)
        kv2 = mock_kv_cache(100 * 1024)

        small_cache.store(tokens1, kv1)
        small_cache.store(tokens2, kv2)

        # Fetch first entry (moves it to end of LRU)
        small_cache.fetch(tokens1)

        # Now tokens2 should be evicted first if we need space
        # Store a large entry to trigger eviction
        big_kv = mock_kv_cache(900 * 1024)
        small_cache.store([7, 8, 9], big_kv)

        # tokens1 should still be there (was recently accessed)
        # tokens2 should be evicted
        assert tokens1 in small_cache or len(small_cache) == 1

    def test_entry_too_large_rejected(self, small_cache, mock_kv_cache):
        # Try to store entry larger than cache limit
        tokens = [1, 2, 3]
        huge_kv = mock_kv_cache(10 * 1024 * 1024)  # 10MB, limit is 1MB

        result = small_cache.store(tokens, huge_kv)
        assert result is False
        assert len(small_cache) == 0

    def test_store_empty_rejected(self, small_cache, mock_kv_cache):
        assert small_cache.store([], mock_kv_cache(100)) is False
        assert small_cache.store([1, 2, 3], []) is False
        assert small_cache.store([1, 2, 3], None) is False

    def test_remove_entry(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)
        assert len(small_cache) == 1

        assert small_cache.remove(tokens) is True
        assert len(small_cache) == 0
        assert small_cache.remove(tokens) is False  # Already removed

    def test_clear(self, small_cache, mock_kv_cache):
        for i in range(3):
            small_cache.store([i], mock_kv_cache(1000))

        assert len(small_cache) == 3
        small_cache.clear()
        assert len(small_cache) == 0
        assert small_cache.memory_usage_mb == 0

    def test_contains(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        assert tokens not in small_cache
        small_cache.store(tokens, mock_kv_cache(1000))
        assert tokens in small_cache

    def test_stats_tracking(self, small_cache, mock_kv_cache):
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens1, kv)
        small_cache.fetch(tokens1)  # Hit
        small_cache.fetch(tokens2)  # Miss

        stats = small_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1

    def test_try_reserve_and_release_memory(self, small_cache):
        assert small_cache.try_reserve_memory(512 * 1024) is True
        assert small_cache.get_stats()["current_memory_mb"] == 0.5

        small_cache.release_reserved_memory(128 * 1024)
        assert small_cache.get_stats()["current_memory_mb"] == 0.38

    def test_try_reserve_memory_denies_over_limit(self, small_cache):
        assert small_cache.try_reserve_memory(2 * 1024 * 1024) is False
        assert small_cache.get_stats()["current_memory_mb"] == 0

    def test_memory_mutations_wait_for_memory_lock(self, small_cache, mock_kv_cache):
        def assert_waits_for_lock(operation):
            result = {}
            errors = []
            small_cache._memory_lock.acquire()

            def run_operation():
                try:
                    result["value"] = operation()
                except Exception as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            thread = threading.Thread(target=run_operation)
            thread.start()
            try:
                time.sleep(0.05)
                assert thread.is_alive()
            finally:
                small_cache._memory_lock.release()
            thread.join(timeout=1)

            assert not thread.is_alive()
            assert errors == []
            return result.get("value")

        assert assert_waits_for_lock(
            lambda: small_cache.store([10], mock_kv_cache(1000))
        )

        small_cache.store([20], mock_kv_cache(1000))
        assert assert_waits_for_lock(lambda: small_cache.remove([20]))

        small_cache.store([30], mock_kv_cache(1000))
        assert_waits_for_lock(small_cache.clear)
        assert len(small_cache) == 0
        assert small_cache.memory_usage_mb == 0

    def test_reset_stats(self, small_cache, mock_kv_cache):
        small_cache.store([1, 2, 3], mock_kv_cache(1000))
        small_cache.fetch([1, 2, 3])
        small_cache.fetch([4, 5, 6])

        small_cache.reset_stats()
        stats = small_cache.get_stats()

        # Stats reset but entry count preserved
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entry_count"] == 1

    def test_duplicate_store_updates_lru(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens, kv)
        initial_len = len(small_cache)

        # Store same tokens again
        small_cache.store(tokens, kv)

        # Should not create duplicate
        assert len(small_cache) == initial_len

    def test_max_entries_limit(self, model, mock_kv_cache):
        # Create cache with low entry limit
        config = MemoryCacheConfig(max_memory_mb=100, max_entries=3)
        cache = MemoryAwarePrefixCache(model, config)

        # Store 5 entries (only 3 should remain)
        for i in range(5):
            cache.store([i], mock_kv_cache(100))

        assert len(cache) <= 3


class TestGetAvailableMemory:
    """Tests for _get_available_memory helper."""

    def test_with_psutil(self):
        try:
            from importlib.util import find_spec

            if find_spec("psutil") is None:
                pytest.skip("psutil not installed")
            mem = _get_available_memory()
            assert mem > 0
        except ImportError:
            pytest.skip("psutil not installed")

    def test_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Should return 0 when psutil not available
            # Note: This test may not work as expected due to import caching
            pass


class TestLockFreeFetchVsRemovalRaces:
    """Regressions for the fetch()-vs-removal races.

    ``fetch()`` is lock-free while eviction / ``remove()`` / ``clear()``
    mutate ``_entries`` and ``_sorted_keys`` in multiple steps, so a fetch
    could observe half-finished removals: a sorted-index key whose entry
    was already popped (``KeyError`` at discovery or at the LRU
    ``move_to_end`` touch), or the live index list shrinking mid-scan
    (``IndexError``).  The fix: removals unindex before popping, fetch
    scans a snapshot of the index, and every dereference tolerates a
    concurrently vanished key by degrading to a miss.
    """

    @staticmethod
    def _make_cache(**cfg):
        cfg.setdefault("max_memory_mb", 64)
        cfg.setdefault("min_prefix_tokens", 1)
        model = MagicMock()
        return MemoryAwarePrefixCache(model, MemoryCacheConfig(**cfg))

    @staticmethod
    def _layer():
        # Array-free layer object: estimator prices it at 0 bytes and
        # store() keeps it as-is, which is all these tests need.
        class _Layer:
            pass

        return [_Layer()]

    def test_stale_index_key_degrades_to_miss_not_crash(self):
        """A key present in _sorted_keys but missing from _entries (the
        mid-removal state a lock-free fetch can observe) must produce a
        miss at every discovery site — exact, prefix, supersequence, and
        LCP — never a KeyError escaping the public fetch() API."""
        cache = self._make_cache()
        assert cache.store([1, 2, 3], self._layer())

        # Simulate the torn state directly: entry gone, index stale.
        del cache._entries[(1, 2, 3)]
        assert (1, 2, 3) in cache._sorted_keys

        # Exact + supersequence discovery.
        got, remaining = cache.fetch([1, 2, 3])
        assert got is None and remaining == [1, 2, 3]
        # Prefix discovery (stale key is a strict prefix of the query).
        got, remaining = cache.fetch([1, 2, 3, 9])
        assert got is None and remaining == [1, 2, 3, 9]
        # Supersequence discovery (query is a strict prefix of stale key).
        got, remaining = cache.fetch([1, 2])
        assert got is None and remaining == [1, 2]

    def test_lru_touch_on_vanished_key_is_noop(self):
        """The best-effort LRU touch must tolerate a key evicted between
        discovery and the touch."""
        cache = self._make_cache()
        cache._touch((9, 9, 9))  # must not raise

    def test_removal_paths_unindex_before_popping_entries(self):
        """Every removal path must remove the key from the sorted index
        BEFORE popping the entry: lock-free fetch() discovers entries
        through the index, so 'advertised but gone' must never be an
        observable state.  Verified by recording the operation order."""
        from collections import OrderedDict

        log = []

        class _LoggingEntries(OrderedDict):
            def pop(self, *args, **kwargs):
                log.append("pop_entry")
                return super().pop(*args, **kwargs)

            def popitem(self, *args, **kwargs):
                log.append("pop_entry")
                return super().popitem(*args, **kwargs)

            def clear(self):
                log.append("pop_entry")
                return super().clear()

        class _LoggingIndex(list):
            def clear(self):
                log.append("unindex")
                return super().clear()

        def _instrument(cache):
            new = _LoggingEntries()
            new.update(cache._entries)
            cache._entries = new
            cache._sorted_keys = _LoggingIndex(cache._sorted_keys)
            original = cache._remove_from_sorted

            def logging_unindex(key):
                log.append("unindex")
                return original(key)

            cache._remove_from_sorted = logging_unindex

        # remove()
        cache = self._make_cache()
        cache.store([1, 2, 3], self._layer())
        _instrument(cache)
        assert cache.remove([1, 2, 3])
        assert log == ["unindex", "pop_entry"]

        # LRU eviction (max_entries=1 forces eviction on second store)
        log.clear()
        cache = self._make_cache(max_entries=1)
        cache.store([1, 2, 3], self._layer())
        _instrument(cache)
        cache.store([4, 5, 6], self._layer())
        assert log[:2] == ["unindex", "pop_entry"]

        # clear()
        log.clear()
        cache = self._make_cache()
        cache.store([1, 2, 3], self._layer())
        _instrument(cache)
        cache.clear()
        assert log == ["unindex", "pop_entry"]

    def test_concurrent_fetch_store_remove_clear_no_escaped_exceptions(self):
        """Integration hammer: concurrent stores, fetches, removes, and
        clears must never let an exception escape the public API — the
        worst permitted outcome of any interleaving is a cache miss."""
        import sys

        cache = self._make_cache(max_entries=8)
        errors = []
        stop = time.monotonic() + 2.0
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-5)

        def storer(base):
            i = 0
            while time.monotonic() < stop:
                i += 1
                key = [base, i % 13] + list(range(i % 7))
                try:
                    cache.store(key, self._layer())
                except Exception as exc:  # pragma: no cover
                    errors.append(("store", exc))

        def fetcher(base):
            i = 0
            while time.monotonic() < stop:
                i += 1
                # Mix of exact, prefix-shaped, and divergent queries.
                query = [base, i % 13] + list(range(i % 9))
                try:
                    cache.fetch(query)
                except Exception as exc:
                    errors.append(("fetch", exc))

        def remover():
            i = 0
            while time.monotonic() < stop:
                i += 1
                try:
                    cache.remove([i % 3, i % 13])
                except Exception as exc:  # pragma: no cover
                    errors.append(("remove", exc))
                if i % 50 == 0:
                    try:
                        cache.clear()
                    except Exception as exc:  # pragma: no cover
                        errors.append(("clear", exc))

        threads = (
            [threading.Thread(target=storer, args=(b,)) for b in range(2)]
            + [threading.Thread(target=fetcher, args=(b,)) for b in range(3)]
            + [threading.Thread(target=remover)]
        )
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old_interval)

        assert errors == [], f"escaped exceptions: {errors[:5]}"

    def test_prefix_subset_eviction_unindexes_before_popping(self):
        """The prefix-subset eviction inside store() must follow the same
        unindex-before-pop order as every other removal path."""
        from collections import OrderedDict

        log = []

        class _LoggingEntries(OrderedDict):
            def pop(self, *args, **kwargs):
                log.append("pop_entry")
                return super().pop(*args, **kwargs)

        cache = self._make_cache()
        cache.store([1, 2, 3], self._layer())

        new = _LoggingEntries()
        new.update(cache._entries)
        cache._entries = new
        original = cache._remove_from_sorted

        def logging_unindex(key):
            log.append("unindex")
            return original(key)

        cache._remove_from_sorted = logging_unindex

        # Storing a supersequence evicts the strict-prefix entry.
        assert cache.store([1, 2, 3, 4], self._layer(), evict_prefixes=True)
        assert "pop_entry" in log, "prefix-subset eviction did not fire"
        assert log.index("unindex") < log.index("pop_entry")

    def test_ghost_longest_prefix_recovers_shorter_prefix(self):
        """If the longest indexed prefix vanished mid-race, a shorter
        prefix that is still fully present must be served — the torn
        window should cost at most the delta, not the whole hit."""
        cache = self._make_cache()
        assert cache.store([1, 2, 3], self._layer())
        assert cache.store([1, 2, 3, 4], self._layer(), evict_prefixes=False)

        # Torn state: longest prefix's entry gone, index stale.
        del cache._entries[(1, 2, 3, 4)]

        got, remaining = cache.fetch([1, 2, 3, 4, 5])
        assert got is not None, "shorter present prefix was not recovered"
        assert remaining == [4, 5]

    def test_save_to_disk_tolerates_concurrent_mutation(self, tmp_path):
        """save_to_disk() must survive the entries dict mutating mid-
        iteration (concurrent store/eviction, or fetch's lock-free LRU
        touch relinking the OrderedDict).  Deterministic: the instrumented
        dict injects one real mutation after the second item is yielded —
        unfixed code lets the resulting RuntimeError escape the public
        API; fixed code snapshots under the lock with a retry, so the
        first attempt absorbs the mutation and the second succeeds."""
        from collections import OrderedDict

        # save_to_disk() returns before iterating when mlx_lm is absent,
        # so the iteration under test only exists with mlx_lm installed.
        pytest.importorskip("mlx_lm")

        cache = self._make_cache()
        for i in range(6):
            cache.store([i, i + 1, i + 2], self._layer())

        class _MutatingEntries(OrderedDict):
            fired = False

            def items(inner):
                it = iter(super().items())

                def gen():
                    for n, kv in enumerate(it):
                        yield kv
                        if n == 1 and not inner.fired:
                            inner.fired = True
                            # Real mutation of the underlying dict —
                            # invalidates every live iterator over it.
                            inner[("injected", "mid", "iteration")] = kv[1]

                return gen()

        new = _MutatingEntries()
        new.update(cache._entries)
        cache._entries = new

        # Must not raise; per-entry safetensors failures are fine (the
        # fake layers aren't persistable), only escaped iteration errors
        # are the bug.
        cache.save_to_disk(str(tmp_path / "snap"))
        assert new.fired, "instrumentation never triggered"
