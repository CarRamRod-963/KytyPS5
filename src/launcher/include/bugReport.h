#ifndef LAUNCHER_INCLUDE_BUG_REPORT_H_
#define LAUNCHER_INCLUDE_BUG_REPORT_H_

#include <QByteArray>
#include <QDialog>
#include <QList>
#include <QPair>
#include <QString>
#include <QStringList>

class QPlainTextEdit;
class QTreeWidget;

// Advanced -> Support -> Export bug report: one zip with logs, settings and system information, made on this PC.
// Nothing is sent: the dialog lists every file (with a preview of what goes in), says that GitHub issues are public,
// saves the zip and then opens the fork's issues page (FORK_REPOSITORY_URL, forkInfo.h) or, before the repository is
// public, the folder of the zip.
namespace BugReport {

// Personal data in the files: the Windows user name (in paths, and as a word when it has 3 or more characters) becomes
// <user>, the computer name <computer>, the PS5 user names of the settings (--user-name, user_name=, and as words)
// <user-name>.
class Scrubber {
public:
	Scrubber();

	void AddUserName(const QString& name);

	[[nodiscard]] QString Scrub(const QString& text) const;

private:
	QStringList m_windows_users;
	QStringList m_computers;
	QStringList m_user_names;
};

struct File {
	QString name;   // path inside the zip ("logs/kyty-20261023-170000.log")
	QString source; // the file it comes from; empty for a generated one
	QString text;   // a generated file's content
};

// Writes a zip (deflate; stored when that is not smaller). False on a write error.
bool WriteZip(const QString& path, const QList<QPair<QString, QByteArray>>& entries);

class Dialog: public QDialog {
	Q_OBJECT

public:
	Dialog(const QList<File>& files, const Scrubber& scrubber, QWidget* parent = nullptr);

	// Writes the zip there without asking (also used by the launcher's screenshot check). False on a write error.
	bool SaveTo(const QString& path);

private:
	void Save();
	void ShowPreview();

	QList<QPair<QString, QByteArray>> m_entries; // name, scrubbed content
	QTreeWidget*                      m_list    = nullptr;
	QPlainTextEdit*                   m_preview = nullptr;
};

} // namespace BugReport

#endif // LAUNCHER_INCLUDE_BUG_REPORT_H_
