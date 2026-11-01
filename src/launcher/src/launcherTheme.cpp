#include "launcherTheme.h"

#include <QApplication>
#include <QDirIterator>
#include <QEvent>
#include <QFile>
#include <QFontDatabase>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>
#include <QPalette>
#include <QRegularExpression>
#include <QStyleFactory>
#include <QTimer>
#include <QVariant>
#include <QWidget>

namespace {

class PaletteChangeFilter final: public QObject {
public:
	explicit PaletteChangeFilter(QApplication& application)
	    : QObject(&application), m_application(application) {}

protected:
	bool eventFilter(QObject* watched, QEvent* event) override {
		if (event->type() == QEvent::ApplicationPaletteChange && !m_refresh_pending) {
			m_refresh_pending = true;
			QTimer::singleShot(0, &m_application, [this] {
				m_application.setStyleSheet(m_application.styleSheet());
				m_refresh_pending = false;
			});
		}

		return QObject::eventFilter(watched, event);
	}

private:
	QApplication& m_application;
	bool          m_refresh_pending = false;
};

struct Tokens {
	QHash<QString, QColor>      colors;   // palette names and semantic names ("text/secondary" and "text-secondary")
	QHash<QString, QJsonObject> types;    // text styles
	QHash<QString, QStringList> fallback; // font family -> fallback families
};

const Tokens& GetTokens() {
	static const Tokens tokens = [] {
		Tokens t;
		QFile  file(QStringLiteral(":/styles/tokens.json"));
		if (!file.open(QIODevice::ReadOnly)) {
			return t;
		}
		const auto root    = QJsonDocument::fromJson(file.readAll()).object();
		const auto palette = root.value(QStringLiteral("palette")).toObject();
		for (auto it = palette.begin(); it != palette.end(); ++it) {
			t.colors.insert(it.key(), QColor(it.value().toString()));
		}
		const auto color = root.value(QStringLiteral("color")).toObject();
		for (auto it = color.begin(); it != color.end(); ++it) {
			const auto value = t.colors.value(it.value().toString());
			t.colors.insert(it.key(), value);
			t.colors.insert(QString(it.key()).replace(QLatin1Char('/'), QLatin1Char('-')), value);
		}
		const auto type = root.value(QStringLiteral("type")).toObject();
		for (auto it = type.begin(); it != type.end(); ++it) {
			t.types.insert(it.key(), it.value().toObject());
		}
		const auto fallback = root.value(QStringLiteral("fallback")).toObject();
		for (auto it = fallback.begin(); it != fallback.end(); ++it) {
			for (const auto& family: it.value().toArray()) {
				t.fallback[it.key()] << family.toString();
			}
		}
		return t;
	}();
	return tokens;
}

// "Family/weight" -> the family name a bundled static font got: Windows may list each weight of a family as its own
// family ("Cormorant Garamond SemiBold").
QHash<QString, QString>& LoadedFamilies() {
	static QHash<QString, QString> families;
	return families;
}

QString FamilyKey(const QString& family, int weight) {
	return family + QLatin1Char('/') + QString::number(weight);
}

// Bundled fonts (:/fonts/*.ttf, *.otf), when present; otherwise the fallbacks of tokens.json are used.
void LoadFonts() {
	const auto& tokens = GetTokens();
	QDirIterator it(QStringLiteral(":/fonts"), {QStringLiteral("*.ttf"), QStringLiteral("*.otf")}, QDir::Files);
	while (it.hasNext()) {
		const int id = QFontDatabase::addApplicationFont(it.next());
		for (const auto& loaded: QFontDatabase::applicationFontFamilies(id)) {
			for (auto design = tokens.fallback.begin(); design != tokens.fallback.end(); ++design) {
				if (!loaded.startsWith(design.key())) {
					continue;
				}
				for (const auto& style: QFontDatabase::styles(loaded)) {
					LoadedFamilies().insert(FamilyKey(design.key(), QFontDatabase::weight(loaded, style)), loaded);
				}
			}
		}
	}
	for (auto it = tokens.fallback.begin(); it != tokens.fallback.end(); ++it) {
		QFont::insertSubstitutions(it.key(), it.value());
	}
}

// @name in launcher.qss -> the token colour (#rrggbb).
QString ResolveStyleSheet(QString style_sheet) {
	static const QRegularExpression token(QStringLiteral("@([a-z][a-z0-9-]*)"));
	const auto&                     tokens = GetTokens();
	QString                         out;
	qsizetype                       last = 0;
	for (auto it = token.globalMatch(style_sheet); it.hasNext();) {
		const auto match = it.next();
		out += style_sheet.mid(last, match.capturedStart() - last);
		const auto name = match.captured(1);
		if (tokens.colors.contains(name)) {
			out += tokens.colors.value(name).name();
		} else {
			qWarning("launcher.qss: unknown token @%s", qPrintable(name));
			out += match.captured(0);
		}
		last = match.capturedEnd();
	}
	out += style_sheet.mid(last);
	return out;
}

QPalette CreatePalette() {
	using LauncherTheme::Color;
	QPalette p;
	p.setColor(QPalette::Window, Color(QStringLiteral("bg/page")));
	p.setColor(QPalette::WindowText, Color(QStringLiteral("text/primary")));
	p.setColor(QPalette::Base, Color(QStringLiteral("bg/field")));
	p.setColor(QPalette::AlternateBase, Color(QStringLiteral("bg/page")));
	p.setColor(QPalette::Text, Color(QStringLiteral("text/primary")));
	p.setColor(QPalette::PlaceholderText, Color(QStringLiteral("text/secondary")));
	p.setColor(QPalette::Button, Color(QStringLiteral("bg/card")));
	p.setColor(QPalette::ButtonText, Color(QStringLiteral("text/primary")));
	p.setColor(QPalette::BrightText, Color(QStringLiteral("text/on-dark-strong")));
	p.setColor(QPalette::Light, Color(QStringLiteral("paper-light")));
	p.setColor(QPalette::Midlight, Color(QStringLiteral("paper")));
	p.setColor(QPalette::Mid, Color(QStringLiteral("border/default")));
	p.setColor(QPalette::Dark, Color(QStringLiteral("stone")));
	p.setColor(QPalette::Shadow, Color(QStringLiteral("ink")));
	// No gold here: gold marks one thing per screen (the Play button or the active page).
	p.setColor(QPalette::Highlight, Color(QStringLiteral("mist")));
	p.setColor(QPalette::HighlightedText, Color(QStringLiteral("text/primary")));
	p.setColor(QPalette::Link, Color(QStringLiteral("text/primary")));
	p.setColor(QPalette::LinkVisited, Color(QStringLiteral("text/secondary")));
	p.setColor(QPalette::ToolTipBase, Color(QStringLiteral("ink")));
	p.setColor(QPalette::ToolTipText, Color(QStringLiteral("paper")));
	for (const auto role: {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
		p.setColor(QPalette::Disabled, role, Color(QStringLiteral("text/secondary")));
	}
	return p;
}

} // namespace

namespace LauncherTheme {

QColor Color(const QString& token) {
	return GetTokens().colors.value(token, QColor(Qt::magenta));
}

QFont TextStyle(const QString& name) {
	const auto  style  = GetTokens().types.value(name);
	const auto  family = style.value(QStringLiteral("family")).toString();
	const auto  size   = style.value(QStringLiteral("size")).toInt(14);
	const int   weight = style.value(QStringLiteral("weight")).toInt(400);
	QStringList families;
	if (const auto loaded = LoadedFamilies().value(FamilyKey(family, weight)); !loaded.isEmpty() && loaded != family) {
		families << loaded;
	}
	families << family << GetTokens().fallback.value(family);
	QFont font(families.first());
	font.setFamilies(families);
	font.setPixelSize(size);
	font.setWeight(static_cast<QFont::Weight>(weight));
	if (const auto tracking = style.value(QStringLiteral("tracking")).toDouble(); tracking != 0.0) {
		font.setLetterSpacing(QFont::AbsoluteSpacing, size * tracking / 100.0);
	}
	if (style.value(QStringLiteral("caps")).toBool()) {
		font.setCapitalization(QFont::AllUppercase);
	}
	return font;
}

void ApplyTextStyles(QWidget* root) {
	for (auto* widget: root->findChildren<QWidget*>()) {
		const auto name = widget->property("textStyle");
		if (name.isValid()) {
			// A style-sheet-polished widget without a font rule takes its parent's font, so the family, size and
			// weight go into a font rule too; tracking and capitals stay from setFont (the rule resolves over it).
			const auto font = TextStyle(name.toString());
			widget->setFont(font);
			widget->setStyleSheet(QStringLiteral("font-family: \"%1\"; font-size: %2px; font-weight: %3;")
			                          .arg(font.families().value(0, font.family()))
			                          .arg(font.pixelSize())
			                          .arg(static_cast<int>(font.weight())));
		}
	}
}

void Initialize(QApplication& application) {
	LoadFonts();
	// Fusion follows the palette and the style sheet on every platform.
	application.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
	application.setPalette(CreatePalette());
	application.setFont(TextStyle(QStringLiteral("Body/Default")));

	QFile style_sheet_file(QStringLiteral(":/styles/launcher.qss"));
	if (style_sheet_file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		application.setStyleSheet(ResolveStyleSheet(QString::fromUtf8(style_sheet_file.readAll())));
		application.installEventFilter(new PaletteChangeFilter(application));
	}
}
} // namespace LauncherTheme
