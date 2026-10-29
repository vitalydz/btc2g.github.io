// Account registrations, traffic estimates and confirmed email subscriptions
// deliberately have different meanings. Missing evidence is never a zero.
export function confirmedSubscriptions() {
  return {
    status: 'unavailable',
    total: null,
    newToday: null,
    reason: 'No confirmed email subscription source is connected.'
  };
}

export function trafficMetrics(row) {
  const firstSeen = row?.first_seen_at;
  if (firstSeen === null || firstSeen === undefined) {
    return {
      status: 'unavailable', visitorsToday: null, viewsToday: null,
      firstRecordedVisitAt: null,
      reason: 'No recorded page visits are available yet.'
    };
  }
  return {
    status: 'available',
    visitorsToday: Number(row.visitors_today),
    viewsToday: Number(row.views_today),
    firstRecordedVisitAt: new Date(Number(firstSeen) * 1000).toISOString(),
    methodology: 'Daily IP + browser hash estimate; not verified people. Script blockers and automated traffic can affect counts.'
  };
}

export function registrationRatio(registrations, visitors) {
  // This is an aggregate ratio, not attributed funnel conversion.
  return Number.isFinite(visitors) && visitors > 0 ? registrations / visitors : null;
}
