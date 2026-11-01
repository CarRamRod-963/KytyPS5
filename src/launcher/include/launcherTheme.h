#ifndef LAUNCHER_THEME_H
#define LAUNCHER_THEME_H

#include <QColor>
#include <QFont>
#include <QString>

class QApplication;
class QWidget;

namespace LauncherTheme {

void Initialize(QApplication& application);

// A palette or semantic colour of styles/tokens.json ("ink", "text/secondary").
QColor Color(const QString& token);

// A text style of styles/tokens.json ("Heading/Title"): family with fallbacks, weight, pixel size, tracking.
QFont TextStyle(const QString& name);

// Sets the font of every child widget that has a "textStyle" property (forms/*.ui).
void ApplyTextStyles(QWidget* root);

} // namespace LauncherTheme

#endif // LAUNCHER_THEME_H
