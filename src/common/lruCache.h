#ifndef EMULATOR_SRC_COMMON_LRUCACHE_H_
#define EMULATOR_SRC_COMMON_LRUCACHE_H_

#include <cstddef>
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
		Item*  next = nullptr;
		Item*  prev = nullptr;
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

	void Touch(size_t id, Tick tick) {
		auto& item = m_items[id];
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
		Detach(m_items[id]);
		m_free.push_back(id);
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

	// The oldest objects at or below `tick` that `eligible` accepts, at most `limit` of them,
	// examining at most `max_visits` objects. Rejected objects do not count against `limit`, so
	// old objects that can never be selected do not hide the evictable ones behind them.
	template <typename Predicate>
	[[nodiscard]] std::vector<Object> SelectBelow(Tick tick, size_t limit, size_t max_visits,
	                                              Predicate&& eligible) const {
		std::vector<Object> selected;
		const auto*         item = m_first;
		for (size_t visits = 0; visits < max_visits && selected.size() < limit; visits++) {
			if (item == nullptr || item->tick > tick) {
				break;
			}
			if (eligible(item->object)) {
				selected.push_back(item->object);
			}
			item = item->next;
		}
		return selected;
	}

private:
	void Attach(Item& item) {
		item.prev = m_last;
		item.next = nullptr;
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
	}

	std::deque<Item>    m_items;
	std::vector<size_t> m_free;
	Item*               m_first = nullptr;
	Item*               m_last  = nullptr;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LRUCACHE_H_
