#ifndef GAME_LIST_TREE_WIDGET_H
#define GAME_LIST_TREE_WIDGET_H

#include <QModelIndex>
#include <QTreeWidget>

// The game list's rows (ConfigurationItem). Play shows them as tiles (GameTileView on this model, sharing its
// selection); the table itself is shown only in --local mode, where the compatibility status and comment are edited.
class GameListTreeWidget: public QTreeWidget {
public:
	explicit GameListTreeWidget(QWidget* parent = nullptr): QTreeWidget(parent) {}

	[[nodiscard]] QTreeWidgetItem* ItemFromIndex(const QModelIndex& index) const { return itemFromIndex(index); }
};

#endif // GAME_LIST_TREE_WIDGET_H
