#ifndef CONFIGURATION_LIST_WIDGET_H
#define CONFIGURATION_LIST_WIDGET_H

#include "configuration.h"

#include <QMap>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <memory>

class ConfigurationItem;
class CompatibilityDatabase;
class QEvent;
class QTreeWidgetItem;
class QPoint;

namespace Ui {
class ConfigurationListWidget;
} // namespace Ui

class ConfigurationListWidget: public QWidget {
	Q_OBJECT

public:
	explicit ConfigurationListWidget(QWidget* parent = nullptr);
	~ConfigurationListWidget() override;

	void SetRunEnabled(bool flag) { m_run_enabled = flag; }
	void SetRuntimeDirectory(const QString& directory) { m_runtime_directory = directory; }

	[[nodiscard]] const ConfigurationItem* GetSelectedItem() const { return m_selected_item; }
	ConfigurationItem*                     GetSelectedItem() { return m_selected_item; }

	[[nodiscard]] const QString& GetSettingsFile() const { return m_settings_file; }
	[[nodiscard]] const QString& GetGlobalControllerColor() const {
		return m_global_info.controller.color;
	}
	[[nodiscard]] std::unique_ptr<Configuration>
	CreateConfiguration(const ConfigurationItem& item) const;

	// The settings the Display and Graphics pages edit: the selected game's own settings when it has them, else the
	// global ones. CommitSettings saves them and refreshes the pages (Select).
	[[nodiscard]] Configuration* EditableConfiguration();
	[[nodiscard]] bool           EditsGameSettings() const;
	void                         CommitSettings();

	[[nodiscard]] int GameCount() const;
	// The PS5 user names of the global and per-game settings (left out of bug reports).
	[[nodiscard]] QStringList UserNames() const;
	// Adds a game folder (a game's own folder or a folder of games), rescans and selects its first game. A folder
	// without a game is not kept; returns false.
	bool AddGameDirectory(const QString& dir);
	void ScanGameDirectory();
	void ViewTrophies();
	void ViewTrophyOverview();
	// Play over a game's darkened picture: paper icons and text on dark glass (tiles, toolbar, search).
	void SetDark(bool dark);

signals:

	void Run();
	void Select();
	void PreviewControllerColor(const QString& color);
	// The toolbar's "Add a game folder" (MainDialog asks for the folder).
	void AddGameFolder();

protected:
	void changeEvent(QEvent* event) override;

public slots:
	void WriteSettings();
	void ReadSettings();

protected slots:

	void edit_configuration();
	void delete_configuartion();
	void edit_global_settings();
	void edit_input_mapping();
	void list_itemDoubleClicked(QTreeWidgetItem* witem, int column);
	void show_context_menu(const QPoint& pos);
	void open_game_folder();
	void remove_save_data();
	void filter_configurations(const QString& text);

private:
	void               SelectItem(QTreeWidgetItem* witem);
	void               ApplyCompatibility();
	void               UpdateToolbarIcons();

	ConfigurationItem*            m_selected_item = nullptr;
	bool                          m_run_enabled   = true;
	bool                          m_dark          = false;
	Ui::ConfigurationListWidget*  m_ui            = nullptr;
	QString                       m_settings_file;
	QString                       m_runtime_directory;
	QStringList                   m_game_dirs;
	QString                       m_last_game; // game_path of the last selected game (selected again at start)
	Configuration                 m_global_info;
	QMap<QString, Configuration*> m_custom_infos;
	CompatibilityDatabase*        m_compatibility = nullptr;
};

#endif // CONFIGURATION_LIST_WIDGET_H
