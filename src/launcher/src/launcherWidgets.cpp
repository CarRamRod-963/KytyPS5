#include "launcherWidgets.h"

#include "launcherTheme.h"

#include <QEnterEvent>
#include <QFontMetrics>
#include <QIcon>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>

namespace {

// The part of the image to draw: the design crop, trimmed around its centre to the target's aspect ratio.
QRectF CoverSource(const QPixmap& art, const QRectF& crop, const QSizeF& target) {
	QRectF source(crop.x() * art.width(), crop.y() * art.height(), crop.width() * art.width(),
	              crop.height() * art.height());
	if (target.isEmpty() || source.isEmpty()) {
		return source;
	}
	const qreal target_aspect = target.width() / target.height();
	const qreal source_aspect = source.width() / source.height();
	const auto  centre        = source.center();
	if (target_aspect > source_aspect) {
		source.setHeight(source.width() / target_aspect);
	} else {
		source.setWidth(source.height() * target_aspect);
	}
	source.moveCenter(centre);
	return source;
}

QColor WithAlpha(QColor color, qreal alpha) {
	color.setAlphaF(static_cast<float>(alpha));
	return color;
}

} // namespace

ArtBackground::ArtBackground(QWidget* parent): QWidget(parent) {
	setAttribute(Qt::WA_StyledBackground, false);
}

void ArtBackground::SetArt(const QString& resource, const QRectF& crop) {
	m_art  = QPixmap(resource);
	m_crop = crop;
	update();
}

void ArtBackground::SetScrim(qreal alpha, qreal stop) {
	m_scrim_alpha = alpha;
	m_scrim_stop  = stop;
	update();
}

void ArtBackground::SetBottomFade(int height) {
	m_bottom_fade = height;
	update();
}

void ArtBackground::SetTopWash(qreal alpha, int height) {
	m_top_wash_alpha  = alpha;
	m_top_wash_height = height;
	update();
}

void ArtBackground::SetPicture(const QPixmap& picture, qreal ink_alpha) {
	m_picture     = picture;
	m_picture_ink = ink_alpha;
	update();
}

void ArtBackground::paintEvent(QPaintEvent* /*event*/) {
	QPainter p(this);
	if (!m_picture.isNull()) {
		const auto ink = LauncherTheme::Color(QStringLiteral("ink"));
		p.fillRect(rect(), ink);
		p.setRenderHint(QPainter::SmoothPixmapTransform);
		p.drawPixmap(QRectF(rect()), m_picture, CoverSource(m_picture, {0.0, 0.0, 1.0, 1.0}, size()));
		p.fillRect(rect(), WithAlpha(ink, m_picture_ink));
		return;
	}
	const auto page = LauncherTheme::Color(QStringLiteral("bg/page"));
	p.fillRect(rect(), page);
	if (!m_art.isNull()) {
		p.setRenderHint(QPainter::SmoothPixmapTransform);
		p.drawPixmap(QRectF(rect()), m_art, CoverSource(m_art, m_crop, size()));
	}
	if (m_scrim_alpha > 0.0 && m_scrim_stop > 0.0) {
		QLinearGradient scrim(0, 0, width(), 0);
		scrim.setColorAt(0.0, WithAlpha(page, m_scrim_alpha));
		scrim.setColorAt(m_scrim_stop, WithAlpha(page, 0.0));
		p.fillRect(rect(), scrim);
	}
	if (m_top_wash_alpha > 0.0 && m_top_wash_height > 0) {
		const QRect     band(0, 0, width(), m_top_wash_height);
		QLinearGradient wash(band.topLeft(), band.bottomLeft());
		wash.setColorAt(0.0, WithAlpha(page, m_top_wash_alpha));
		wash.setColorAt(1.0, WithAlpha(page, 0.0));
		p.fillRect(band, wash);
	}
	if (m_bottom_fade > 0) {
		const QRect     band(0, height() - m_bottom_fade, width(), m_bottom_fade);
		QLinearGradient fade(band.topLeft(), band.bottomLeft());
		fade.setColorAt(0.0, WithAlpha(page, 0.0));
		fade.setColorAt(1.0, page);
		p.fillRect(band, fade);
	}
}

ArtButton::ArtButton(QWidget* parent): QPushButton(parent) {
	setAttribute(Qt::WA_Hover);
	setCursor(Qt::PointingHandCursor);
}

void ArtButton::SetArt(const QString& resource, const QRectF& crop) {
	m_art  = QPixmap(resource);
	m_crop = crop;
	update();
}

QSize ArtButton::sizeHint() const {
	constexpr int PADDING = 16;
	const int     width   = QFontMetrics(font()).horizontalAdvance(text()) + 2 * PADDING;
	return {std::max(width, minimumWidth()), std::max(40, minimumHeight())};
}

void ArtButton::enterEvent(QEnterEvent* event) {
	update();
	QPushButton::enterEvent(event);
}

void ArtButton::leaveEvent(QEvent* event) {
	update();
	QPushButton::leaveEvent(event);
}

void ArtButton::paintEvent(QPaintEvent* /*event*/) {
	constexpr qreal RADIUS = 6.0; // radius/sm
	constexpr qreal BORDER = 1.5;

	const bool enabled = isEnabled();
	const bool hover   = enabled && underMouse();
	const bool pressed = enabled && isDown();

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	p.setRenderHint(QPainter::SmoothPixmapTransform);
	if (!enabled) {
		p.setOpacity(0.4);
	}

	QPainterPath shape;
	shape.addRoundedRect(QRectF(rect()), RADIUS, RADIUS);
	p.save();
	p.setClipPath(shape);
	const auto ink = LauncherTheme::Color(QStringLiteral("bg/bar"));
	if (!m_art.isNull()) {
		// Figma stretches the same image region into every button size.
		const QRectF source(m_crop.x() * m_art.width(), m_crop.y() * m_art.height(), m_crop.width() * m_art.width(),
		                    m_crop.height() * m_art.height());
		p.drawPixmap(QRectF(rect()), m_art, source);
	}
	p.fillRect(rect(), WithAlpha(ink, pressed ? 0.70 : (hover ? 0.48 : 0.60)));
	p.restore();

	const auto frame = LauncherTheme::Color(hover || pressed ? QStringLiteral("accent/default") : QStringLiteral("accent/hover"));
	p.setPen(QPen(frame, BORDER));
	p.setBrush(Qt::NoBrush);
	p.drawRoundedRect(QRectF(rect()).adjusted(BORDER / 2, BORDER / 2, -BORDER / 2, -BORDER / 2), RADIUS - BORDER / 2,
	                  RADIUS - BORDER / 2);

	p.setPen(LauncherTheme::Color(hover || pressed ? QStringLiteral("text/on-dark-strong") : QStringLiteral("text/on-dark")));
	p.setFont(font());
	p.drawText(rect(), Qt::AlignCenter, text());
}

NavButton::NavButton(QWidget* parent): QPushButton(parent) {
	setCheckable(true);
	setAttribute(Qt::WA_Hover);
	setCursor(Qt::PointingHandCursor);
}

void NavButton::SetAccent(bool accent) {
	m_accent = accent;
	update();
}

void NavButton::enterEvent(QEnterEvent* event) {
	update();
	QPushButton::enterEvent(event);
}

void NavButton::leaveEvent(QEvent* event) {
	update();
	QPushButton::leaveEvent(event);
}

void NavButton::paintEvent(QPaintEvent* /*event*/) {
	constexpr int PADDING_LEFT = 24; // space/24
	constexpr int DOT          = 6;
	constexpr int GAP          = 12; // space/12
	constexpr int MARKER       = 2;

	const bool active = isChecked();

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	if (active) {
		p.fillRect(rect(), LauncherTheme::Color(QStringLiteral("bg/sidebar-active")));
		p.fillRect(QRect(0, 0, MARKER, height()),
		           LauncherTheme::Color(m_accent ? QStringLiteral("accent/default") : QStringLiteral("nav/bar")));
	} else if (underMouse()) {
		p.fillRect(rect(), WithAlpha(LauncherTheme::Color(QStringLiteral("bg/sidebar-active")), 0.5));
	}

	const auto dot = active ? (m_accent ? QStringLiteral("accent/default") : QStringLiteral("nav/dot-active"))
	                        : QStringLiteral("nav/dot");
	p.setPen(Qt::NoPen);
	p.setBrush(LauncherTheme::Color(dot));
	p.drawEllipse(QRectF(PADDING_LEFT, (height() - DOT) / 2.0, DOT, DOT));

	p.setPen(LauncherTheme::Color(active ? QStringLiteral("text/on-dark") : QStringLiteral("text/on-dark-muted")));
	p.setFont(font());
	p.drawText(rect().adjusted(PADDING_LEFT + DOT + GAP, 0, 0, 0), Qt::AlignVCenter | Qt::AlignLeft, text());
}

StatusIcon::StatusIcon(QWidget* parent): QWidget(parent) {
	setFixedSize(20, 20);
}

void StatusIcon::SetStatus(Status status) {
	m_status = status;
	update();
}

void StatusIcon::paintEvent(QPaintEvent* /*event*/) {
	constexpr int GLYPH = 12;

	auto color = QStringLiteral("status/success");
	auto glyph = QStringLiteral(":/icons/status-ok-glyph.svg");
	if (m_status == Status::Warn) {
		color = QStringLiteral("status/warning");
		glyph = QStringLiteral(":/icons/status-warn-glyph.svg");
	} else if (m_status == Status::Error) {
		color = QStringLiteral("status/error");
		glyph = QStringLiteral(":/icons/status-error-glyph.svg");
	}

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	p.setPen(Qt::NoPen);
	p.setBrush(LauncherTheme::Color(color));
	p.drawEllipse(QRectF(rect()));
	QIcon(glyph).paint(&p, QRect((width() - GLYPH) / 2, (height() - GLYPH) / 2, GLYPH, GLYPH));
}
