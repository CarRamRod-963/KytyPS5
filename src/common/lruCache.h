#ifndef EMULATOR_SRC_COMMON_LRUCACHE_H_
#define EMULATOR_SRC_COMMON_LRUCACHE_H_

#include "common/config.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <type_traits>
#include <utility>
#include <vector>

namespace Common {

template <typename Object, typename Tick>
class LeastRecentlyUsedCache {
	struct Item {
		Object object {};
		Tick   tick {};
		Item*  next   = nullptr;
		Item*  prev   = nullptr;
		bool   linked = false;
	};

public:
	[[nodiscard]] size_t Insert(Object object, Tick tick) {
		size_t id = 0;
		if (m_free.empty()) {
			id = m_items.size();
			m_items.emplace_back();
		} else {
			id = m_free.back();
			m_free.pop_back();
		}
		auto& item  = m_items[id];
		item.object = std::move(object);
		item.tick   = tick;
		Attach(item);
		return id;
	}

	// A freed item is not in the list: touching it would link it again while it waits for reuse.
	void Touch(size_t id, Tick tick) {
		auto& item = m_items[id];
		if (!item.linked) {
			NoteStale("touch");
			return;
		}
		if (item.tick >= tick) {
			return;
		}
		item.tick = tick;
		if (&item != m_last) {
			Detach(item);
			Attach(item);
		}
	}

	void Free(size_t id) {
		auto& item = m_items[id];
		if (!item.linked) {
			NoteStale("free");
			return;
		}
		Detach(item);
		m_free.push_back(id);
	}

	// Touches and frees of items that were not in the list (ignored, see NoteStale).
	[[nodiscard]] size_t StaleOperations() const noexcept { return m_stale_operations; }

	// The list runs from the oldest to the newest item through every live item exactly once,
	// and no freed item is in it (for tests).
	[[nodiscard]] bool IsConsistent() const {
		const size_t live  = m_items.size() - m_free.size();
		size_t       count = 0;
		const Item*  prev  = nullptr;
		for (const auto* item = m_first; item != nullptr; item = item->next) {
			if (!item->linked || item->prev != prev || ++count > live) {
				return false;
			}
			prev = item;
		}
		if (prev != m_last || count != live) {
			return false;
		}
		for (const auto id: m_free) {
			if (m_items[id].linked) {
				return false;
			}
		}
		return true;
	}

	template <typename Function>
	void ForEachItemBelow(Tick tick, Function&& function) {
		constexpr bool ReturnsBool = std::is_same_v<std::invoke_result_t<Function, Object>, bool>;
		for (auto* item = m_first; item != nullptr;) {
			if (item->tick > tick) {
				return;
			}
			auto* next = item->next;
			if constexpr (ReturnsBool) {
				if (function(item->object)) {
					return;
				}
			} else {
				function(item->object);
			}
			item = next;
		}
	}

private:
	// A touch or free of an item that is not in the list comes from a holder of a stale id.
	// Release builds ignore it; debug builds stop so that the holder is found (tests that
	// exercise stale ids define KYTY_LRU_CACHE_ALLOW_STALE).
	void NoteStale([[maybe_unused]] const char* operation) {
		m_stale_operations++;
#if KYTY_BUILD == KYTY_BUILD_DEBUG && !defined(KYTY_LRU_CACHE_ALLOW_STALE)
		std::fprintf(stderr, "LeastRecentlyUsedCache: %s of an item that is not in the list\n",
		             operation);
		std::abort();
#endif
	}

	void Attach(Item& item) {
		item.prev   = m_last;
		item.next   = nullptr;
		item.linked = true;
		if (m_last != nullptr) {
			m_last->next = &item;
		} else {
			m_first = &item;
		}
		m_last = &item;
	}

	void Detach(Item& item) {
		if (item.prev != nullptr) {
			item.prev->next = item.next;
		} else {
			m_first = item.next;
		}
		if (item.next != nullptr) {
			item.next->prev = item.prev;
		} else {
			m_last = item.prev;
		}
		item.linked = false;
	}

	std::deque<Item>    m_items;
	std::vector<size_t> m_free;
	Item*               m_first = nullptr;
	Item*               m_last  = nullptr;
	size_t              m_stale_operations = 0;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LRUCACHE_H_
