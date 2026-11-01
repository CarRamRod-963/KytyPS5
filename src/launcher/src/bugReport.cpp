#include "bugReport.h"

#include "forkInfo.h"
#include "launcherTheme.h"

#include <QDataStream>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QStandardPaths>
#include <QSysInfo>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <array>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

namespace BugReport {

namespace {

// A log longer than this goes in as its last part (the end of a run is where its error is).
constexpr qint64 MAX_FILE_BYTES = 32LL * 1024 * 1024;
// The preview shows the start of a file.
constexpr int PREVIEW_CHARS = 200000;

void AddUnique(QStringList* list, const QString& value) {
	const auto trimmed = value.trimmed();
	if (!trimmed.isEmpty() && !list->contains(trimmed, Qt::CaseInsensitive)) {
		list->append(trimmed);
	}
}

// Longest first, so a name that contains another is replaced whole.
void SortLongestFirst(QStringList* list) {
	std::sort(list->begin(), list->end(), [](const QString& a, const QString& b) { return a.size() > b.size(); });
}

QString WordPattern(const QString& word) {
	return QStringLiteral("(?<![\\p{L}\\p{N}_])") + QRegularExpression::escape(word) +
	       QStringLiteral("(?![\\p{L}\\p{N}_])");
}

QString ShortHomeName() {
#if defined(_WIN32)
	// The 8.3 form of the profile folder ("RAFAUR~1"), which temporary paths use.
	const auto home = QDir::toNativeSeparators(QDir::homePath()).toStdWString();
	std::array<wchar_t, MAX_PATH> buffer {};
	const auto length = GetShortPathNameW(home.c_str(), buffer.data(), static_cast<DWORD>(buffer.size()));
	if (length > 0 && length < buffer.size()) {
		return QDir(QDir::fromNativeSeparators(QString::fromWCharArray(buffer.data(), static_cast<int>(length))))
		    .dirName();
	}
#endif
	return {};
}

QByteArray ReadForReport(const QString& path) {
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly)) {
		return QObject::tr("(could not read %1: %2)\n").arg(path, file.errorString()).toUtf8();
	}
	if (file.size() <= MAX_FILE_BYTES) {
		return file.readAll();
	}
	file.seek(file.size() - MAX_FILE_BYTES);
	return QObject::tr("(the last %1 MB of %2 MB)\n")
	           .arg(MAX_FILE_BYTES / (1024 * 1024))
	           .arg(file.size() / (1024 * 1024))
	           .toUtf8() +
	       file.readAll();
}

// ---- zip ----

uint32_t Crc32(const QByteArray& data) {
	static const auto table = [] {
		std::array<uint32_t, 256> t {};
		for (uint32_t i = 0; i < 256; i++) {
			uint32_t c = i;
			for (int k = 0; k < 8; k++) {
				c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
			}
			t[i] = c;
		}
		return t;
	}();
	uint32_t crc = 0xFFFFFFFFU;
	for (const char byte: data) {
		crc = table[(crc ^ static_cast<uint8_t>(byte)) & 0xFFU] ^ (crc >> 8U);
	}
	return crc ^ 0xFFFFFFFFU;
}

// Raw deflate data: qCompress gives a 4-byte length, then a zlib stream (2-byte header, deflate, 4-byte Adler-32).
QByteArray RawDeflate(const QByteArray& data) {
	if (data.isEmpty()) {
		return {};
	}
	const auto zlib = qCompress(data, 9);
	return zlib.size() > 10 ? zlib.mid(6, zlib.size() - 10) : QByteArray();
}

} // namespace

Scrubber::Scrubber() {
	AddUnique(&m_windows_users, qEnvironmentVariable("USERNAME"));
	AddUnique(&m_windows_users, qEnvironmentVariable("USER"));
	AddUnique(&m_windows_users, QDir(QDir::homePath()).dirName());
	AddUnique(&m_windows_users, ShortHomeName());
	AddUnique(&m_computers, qEnvironmentVariable("COMPUTERNAME"));
	AddUnique(&m_computers, QSysInfo::machineHostName());
	SortLongestFirst(&m_windows_users);
	SortLongestFirst(&m_computers);
}

void Scrubber::AddUserName(const QString& name) {
	// "Kyty" is the default PS5 user name and part of the emulator's name: not personal.
	if (name.trimmed().compare(QStringLiteral("Kyty"), Qt::CaseInsensitive) != 0) {
		AddUnique(&m_user_names, name);
		SortLongestFirst(&m_user_names);
	}
}

QString Scrubber::Scrub(const QString& text) const {
	static const QRegularExpression user_name_arg(QStringLiteral("(--user-name\"?\\s+)(\"[^\"\\r\\n]*\"|\\S+)"));
	static const QRegularExpression user_name_key(QStringLiteral("^(\\s*(?:\\d+\\\\)?user_name\\s*=).*$"),
	                                              QRegularExpression::MultilineOption);
	auto out = text;
	out.replace(user_name_arg, QStringLiteral("\\1<user-name>"));
	out.replace(user_name_key, QStringLiteral("\\1<user-name>"));
	const auto ci = QRegularExpression::CaseInsensitiveOption;
	for (const auto& user: m_windows_users) {
		out.replace(QRegularExpression(QStringLiteral("([\\\\/](?:Users|home|Documents and Settings)[\\\\/])") +
		                                   QRegularExpression::escape(user) + QStringLiteral("(?![\\p{L}\\p{N}_])"),
		                               ci),
		            QStringLiteral("\\1<user>"));
		if (user.size() >= 3) {
			out.replace(QRegularExpression(WordPattern(user), ci), QStringLiteral("<user>"));
		}
	}
	for (const auto& computer: m_computers) {
		if (computer.size() >= 3) {
			out.replace(QRegularExpression(WordPattern(computer), ci), QStringLiteral("<computer>"));
		}
	}
	for (const auto& name: m_user_names) {
		out.replace(QRegularExpression(WordPattern(name), ci), QStringLiteral("<user-name>"));
	}
	return out;
}

bool WriteZip(const QString& path, const QList<QPair<QString, QByteArray>>& entries) {
	QSaveFile file(path);
	if (!file.open(QIODevice::WriteOnly)) {
		return false;
	}
	QDataStream out(&file);
	out.setByteOrder(QDataStream::LittleEndian);

	const auto now      = QDateTime::currentDateTime();
	const auto dos_time = static_cast<quint16>((now.time().hour() << 11) | (now.time().minute() << 5) |
	                                           (now.time().second() / 2));
	const auto dos_date = static_cast<quint16>(((now.date().year() - 1980) << 9) | (now.date().month() << 5) |
	                                           now.date().day());
	constexpr quint16 VERSION = 20;     // 2.0: deflate
	constexpr quint16 UTF8    = 0x0800; // file names in UTF-8

	struct Central {
		QByteArray name;
		quint16    method = 0;
		quint32    crc    = 0;
		quint32    packed = 0;
		quint32    size   = 0;
		quint32    offset = 0;
	};
	QList<Central> central;
	for (const auto& [name, data]: entries) {
		Central entry;
		entry.name   = name.toUtf8();
		entry.crc    = Crc32(data);
		entry.size   = static_cast<quint32>(data.size());
		entry.offset = static_cast<quint32>(file.pos());
		auto packed  = RawDeflate(data);
		if (!packed.isEmpty() && packed.size() < data.size()) {
			entry.method = 8;
		} else {
			packed = data;
		}
		entry.packed = static_cast<quint32>(packed.size());
		out << quint32(0x04034B50) << VERSION << UTF8 << entry.method << dos_time << dos_date << entry.crc
		    << entry.packed << entry.size << static_cast<quint16>(entry.name.size()) << quint16(0);
		out.writeRawData(entry.name.constData(), static_cast<int>(entry.name.size()));
		out.writeRawData(packed.constData(), static_cast<int>(packed.size()));
		central.append(entry);
	}
	const auto directory = static_cast<quint32>(file.pos());
	for (const auto& entry: central) {
		out << quint32(0x02014B50) << VERSION << VERSION << UTF8 << entry.method << dos_time << dos_date << entry.crc
		    << entry.packed << entry.size << static_cast<quint16>(entry.name.size()) << quint16(0) << quint16(0)
		    << quint16(0) << quint16(0) << quint32(0) << entry.offset;
		out.writeRawData(entry.name.constData(), static_cast<int>(entry.name.size()));
	}
	const auto directory_size = static_cast<quint32>(file.pos()) - directory;
	out << quint32(0x06054B50) << quint16(0) << quint16(0) << static_cast<quint16>(central.size())
	    << static_cast<quint16>(central.size()) << directory_size << directory << quint16(0);
	return out.status() == QDataStream::Ok && file.commit();
}

Dialog::Dialog(const QList<File>& files, const Scrubber& scrubber, QWidget* parent): QDialog(parent) {
	setObjectName(QStringLiteral("bug_report_dialog"));
	setWindowTitle(tr("Export bug report"));
	resize(880, 640);

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(24, 24, 24, 24);
	layout->setSpacing(16);

	auto* title = new QLabel(tr("Export bug report"), this);
	title->setProperty("textStyle", QStringLiteral("Heading/Section"));
	layout->addWidget(title);

	auto* lead = new QLabel(tr("Nothing is sent: the zip is saved on this PC and you attach it to a GitHub issue "
	                           "yourself. GitHub issues are public: anyone can read the files you attach. These "
	                           "files go into the zip:"),
	                        this);
	lead->setWordWrap(true);
	lead->setProperty("textStyle", QStringLiteral("Body/Default"));
	layout->addWidget(lead);

	auto* note = new QLabel(tr("Personal names are replaced before anything is packed: your Windows user name with "
	                           "<user>, the computer name with <computer>, the PS5 user name with <user-name>. Select "
	                           "a file to see exactly what goes in."),
	                        this);
	note->setTextFormat(Qt::PlainText);
	note->setWordWrap(true);
	note->setProperty("textStyle", QStringLiteral("Body/Description"));
	note->setProperty("tone", QStringLiteral("secondary"));
	layout->addWidget(note);

	m_list = new QTreeWidget(this);
	m_list->setObjectName(QStringLiteral("bug_report_files"));
	m_list->setHeaderLabels({tr("File in the zip"), tr("Size"), tr("From")});
	m_list->setRootIsDecorated(false);
	m_list->setUniformRowHeights(true);
	m_list->setProperty("textStyle", QStringLiteral("Body/Default"));
	layout->addWidget(m_list, 2);

	m_preview = new QPlainTextEdit(this);
	m_preview->setObjectName(QStringLiteral("bug_report_preview"));
	m_preview->setReadOnly(true);
	m_preview->setLineWrapMode(QPlainTextEdit::NoWrap);
	m_preview->setProperty("textStyle", QStringLiteral("Mono/Path"));
	layout->addWidget(m_preview, 3);

	auto* buttons = new QHBoxLayout;
	buttons->setSpacing(8);
	buttons->addStretch(1);
	auto* cancel = new QPushButton(tr("Cancel"), this);
	auto* save   = new QPushButton(tr("Save zip..."), this);
	for (auto* button: {cancel, save}) {
		button->setProperty("kind", QStringLiteral("secondary"));
		button->setProperty("textStyle", QStringLiteral("Label/Button"));
		buttons->addWidget(button);
	}
	save->setDefault(true);
	layout->addLayout(buttons);

	for (const auto& file: files) {
		const auto data = file.source.isEmpty() ? file.text.toUtf8() : ReadForReport(file.source);
		const auto text = scrubber.Scrub(QString::fromUtf8(data)).toUtf8();
		m_entries.append({file.name, text});
		auto* item = new QTreeWidgetItem(m_list);
		item->setText(0, file.name);
		item->setText(1, QLocale(QLocale::English).formattedDataSize(text.size(), 1, QLocale::DataSizeTraditionalFormat));
		item->setText(2, file.source.isEmpty() ? tr("made now") : scrubber.Scrub(QDir::toNativeSeparators(file.source)));
	}
	m_list->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	m_list->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);

	connect(m_list, &QTreeWidget::currentItemChanged, this, &Dialog::ShowPreview);
	connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
	connect(save, &QPushButton::clicked, this, &Dialog::Save);
	if (m_list->topLevelItemCount() > 0) {
		m_list->setCurrentItem(m_list->topLevelItem(0));
	}
	LauncherTheme::ApplyTextStyles(this);
}

void Dialog::ShowPreview() {
	const int index = m_list->indexOfTopLevelItem(m_list->currentItem());
	if (index < 0 || index >= m_entries.size()) {
		m_preview->clear();
		return;
	}
	auto text = QString::fromUtf8(m_entries.at(index).second);
	if (text.size() > PREVIEW_CHARS) {
		text = text.left(PREVIEW_CHARS) + tr("\n... (the preview shows the start; the whole file goes into the zip)");
	}
	m_preview->setPlainText(text);
}

bool Dialog::SaveTo(const QString& path) {
	return WriteZip(path, m_entries);
}

void Dialog::Save() {
	auto folder = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
	if (folder.isEmpty()) {
		folder = QDir::homePath();
	}
	const auto suggested = QDir(folder).filePath(
	    QStringLiteral("kyty-bug-report-%1.zip").arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmm"))));
	const auto path = QFileDialog::getSaveFileName(this, tr("Save bug report"), suggested, tr("Zip files (*.zip)"));
	if (path.isEmpty()) {
		return;
	}
	if (!SaveTo(path)) {
		QMessageBox::warning(this, tr("Export bug report"), tr("Could not write %1.").arg(path));
		return;
	}
	accept();

	const QString repository = QString::fromLatin1(FORK_REPOSITORY_URL);
	QMessageBox box(parentWidget());
	box.setWindowTitle(tr("Export bug report"));
	box.setIcon(QMessageBox::Information);
	auto* show_folder = box.addButton(tr("Show folder"), QMessageBox::ActionRole);
	QPushButton* open_issues = nullptr;
	if (repository.isEmpty()) {
		box.setText(tr("Saved %1.\n\nThe fork's GitHub page is not public yet: keep the zip until it is.")
		                .arg(QDir::toNativeSeparators(path)));
	} else {
		box.setText(tr("Saved %1.\n\nOpen a new GitHub issue, describe what happened and attach the zip. The issue "
		               "and the zip are public.")
		                .arg(QDir::toNativeSeparators(path)));
		open_issues = box.addButton(tr("Open GitHub Issues"), QMessageBox::AcceptRole);
	}
	box.addButton(QMessageBox::Close);
	box.exec();
	if (box.clickedButton() == show_folder) {
		QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(path).absolutePath()));
	} else if (open_issues != nullptr && box.clickedButton() == open_issues) {
		QDesktopServices::openUrl(QUrl(repository + QStringLiteral("/issues/new")));
	}
}

} // namespace BugReport
